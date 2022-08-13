import logging
import os
import time
from collections import Counter

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, GuidedBackpropReLUModel
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

CUDA_DEVICE_ORDER='PCI_BUS_ID'
DEVICE_NR = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NR
logging.basicConfig(level=logging.DEBUG)

from gan_inference import load_gan_model
from dataset import (DukeDataset, OCTDataset, RESCDataset)
from refine_pseudo_label import refine_input_by_cam, get_pseudo_label
from models import MultiTaskModel, CAM_Net, Up_Sample
from utils import CAMGeneratorAndSave, OrgLabels, diff_map_for_att, save_models, save_tensorboard
from metrics import calculate_classification_metrics, calculate_roc
from options import Configs
from metrics import scores, record_score

def network_class(args):
    if args.backbone == "resnet18":
        print("Backbone: ResNet18")
        backbone = models.resnet18(pretrained=True)
    elif args.backbone == "vgg16":
        print("Backbone: VGG16")
        backbone = models.vgg16(pretrained=True)
    elif args.backbone == "resnet50":
        print("Backbone: ResNet50")
        backbone = models.resnet50(pretrained=True)
    elif args.backbone == "resnet101":
        print("Backbone: ResNet101")
        backbone = models.resnet101(pretrained=True)
    else:
        raise NotImplementedError("No backbone found for '{}'".format(args.backbone))   
    return backbone

def dsc_loss(y_pred, y_true, varepsilon=1.e-8):
    epsilon = 1.e-8
    y_true = y_true.float()
    y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

    numerator = 2 * (y_true * y_pred * (1-y_pred)).sum() + varepsilon
    denominator = (y_true + y_pred * (1-y_pred)).sum() + varepsilon

    return 1 - numerator / denominator


class Train():
    def __init__(self, is_inference=False):
        self.args = Configs().parse(is_inference)
        if not is_inference:
            self.tb = SummaryWriter('runs_512/{}'.format(self.args.save_folder[8:]))
        self.device = self.args.device 
        if self.device == "cuda":
            print("Number of GPUs: ", torch.cuda.device_count(), "Device Nbr: ", DEVICE_NR)
            
        # torch.manual_seed(42)
        if 'our_dataset' in self.args.root_dirs:
            self.dataset_train = OCTDataset(self.args, data_type='train')
            self.dataset_test = OCTDataset(self.args, data_type='test')
        elif 'RESC' in self.args.root_dirs:
            self.dataset_train = RESCDataset(self.args, data_type='train')
            self.dataset_test = RESCDataset(self.args, data_type='test')
        elif 'BOE' in self.args.root_dirs:
            self.dataset_train = DukeDataset(self.args, data_type='train')
            self.dataset_test = DukeDataset(self.args, data_type='test')
        self.sigmoid = nn.Sigmoid()

        self.backbone = network_class(self.args)
        self.num_class = len(OrgLabels)
        self.num_input_channel = self.dataset_train[0]['image'].shape[0] * 2 if self.args.input_gan else self.dataset_train[0]['image'].shape[0]
        self.num_of_epochs = self.args.num_iteration + self.args.n_epochs
        self.gan_pretrained = False
        with torch.no_grad():
            if self.args.input_gan:
                path = "{}/netG.pth".format(self.args.model_gan)
                self.gan_pretrained = load_gan_model(path)
                print(f' Loaded Pretained GAN weights from {path}.')
        self.transform_norml = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.w_ce = self.args.w_ce
        self.w_dice = self.args.w_dice
        
    def cam_loss(self, y_pred, y_true):
        loss_ce = self.loss_cam(y_pred, y_true)
        loss_dice = dsc_loss(y_pred, y_true)
        return  self.w_ce * loss_ce + self.w_dice * loss_dice  
    
    def train_parameters(self):
        shared_model = MultiTaskModel(self.backbone, self.num_input_channel)
        cam_model = CAM_Net(shared_model, self.num_class, self.args.backbone)
        seg_model = Up_Sample(shared_model, self.num_class, self.args.backbone)
        
        shared_model = nn.DataParallel(shared_model)
        cam_model = nn.DataParallel(cam_model)
        seg_model = nn.DataParallel(seg_model)

        if self.args.continue_train:
            reload_epoch = 'best' if self.args.cp_epoch == 'best' else 40
            print('Loading pretrained model from checkpoint {0}/weights/{1}.pwf'.format(self.args.check_point, reload_epoch))
            checkpoint = torch.load('{0}/weights/{1}.pwf'.format(self.args.check_point, reload_epoch))   
            cam_model.load_state_dict(checkpoint['state_dict'])
        
        self.cam_model = cam_model.to(self.device)
        self.seg_model = seg_model.to(self.device)

        self.cam_optimizer = optim.SGD(cam_model.parameters(), lr=self.args.lr, momentum=0.9)
        self.seg_optimizer = optim.Adam(seg_model.parameters(), lr=self.args.lr)

        self.cam_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.cam_optimizer, step_size=self.args.lr_schedule['step'], gamma=self.args.lr_schedule['gamma'])
        self.loss_cam = nn.BCEWithLogitsLoss() #no need sigmoid for this logits loss
        self.loss_seg = nn.CrossEntropyLoss()        
        
        self.target_layers = [self.cam_model.module.multi_task_model.base_model[-1][-1]] # .module. if use dataparallel
        self.cam = GradCAM(model=self.cam_model, use_cuda=self.device, target_layers=self.target_layers)
        self.CAMGenerationModule = CAMGeneratorAndSave(opts=self.args, cam=self.cam)
        
    def valid_once(self, epoch):
        self.cam_model.eval()
        self.seg_model.eval()
        total_acc_val = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
        total_cls_loss_val, total_seg_loss_val = 0, 0
        gt_list = torch.empty(0,len(OrgLabels)).to(self.device)
        pred_list = torch.empty(0,len(OrgLabels)).to(self.device)
        
        self.CAMGenerationModule.set_epoch(epoch)
        gt_img_list, cam_img_list = [], []
        tensor_for_att = None
        for batch, data in tqdm(enumerate(self.testloader), total=len(self.testloader)):
            image, labels, mask = data["image"].to(self.device), data["labels"].to(self.device), data['mask'].to(self.device)        
            updated_image = image.clone()
            with torch.no_grad():
                if self.args.input_gan:
                    gan_inputs = self.transform_norml(updated_image)
                    healthy_img = self.gan_pretrained.inference(gan_inputs)
                    if self.args.att_module:
                        tensor_for_att = diff_map_for_att(updated_image, healthy_img, mask)
                    updated_image = torch.cat((image, healthy_img), dim=1)

                if (epoch + 1) > self.args.n_epochs or self.args.continue_train:
                    updated_image = refine_input_by_cam(self.args, self.cam_model, updated_image, mask, self.cam)
                outputs = self.cam_model(updated_image, tensor_for_att)
                sig_prediction = self.sigmoid(outputs)
                
            if self.args.segmentation < epoch + 1:
                outputs_for_seg = sig_prediction.clone()
                params_seg = {'inputs': data, 'batch_preds': outputs_for_seg, 'refined': updated_image}
                pseudo_label = get_pseudo_label(params_seg, self.cam) # (batch, w, h) pixel value in (0, nb_class - 1)
                seg_outputs = self.seg_model(updated_image) #(batch, nb_class, w, h), include 0 as background
                seg_loss = self.loss_seg(seg_outputs, pseudo_label.to(self.device))
                total_seg_loss_val += seg_loss.cpu().item()
            params = {'inputs': data, 'batch_preds': sig_prediction, 'refined': updated_image}
            gt_res, pred_res = self.CAMGenerationModule.get_cam_and_save(params)  
            gt_img_list += gt_res
            cam_img_list += pred_res
            
            with torch.no_grad():
                clf_loss = self.cam_loss(outputs, labels)
                total_cls_loss_val += clf_loss.cpu().item()
                batch_accuracies_metrics = calculate_classification_metrics(sig_prediction, labels)
                total_acc_val += Counter(batch_accuracies_metrics)
                gt_list = torch.cat((gt_list, labels))
                pred_list = torch.cat((pred_list, sig_prediction))
        
        score = scores(gt_img_list, cam_img_list, n_class=3)
        print(score)
        record_score(score, 'resc')
        
        # Print accuracy
        valid_acc_epoch, valid_loss_epoch, valid_seg_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc_val.items()}, total_cls_loss_val / (batch + 1), total_seg_loss_val / (batch+1)
        roc_avg, roc_class = calculate_roc(pred_list, gt_list)
        print('- Val loss:', valid_loss_epoch, '- Seg loss: ', valid_seg_loss_epoch,'- Val ROC:', roc_avg,  "- ROC per class:", roc_class, "- Val acc:", valid_acc_epoch)
        return valid_loss_epoch, valid_acc_epoch, roc_avg, roc_class

    def train_once(self, epoch):
        self.cam_model.train()
        self.seg_model.train()
        tensor_for_att = None
        
        total_acc = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
        total_cls_loss, total_seg_loss = 0, 0
        gt_list = torch.empty(0,len(OrgLabels)).to(self.device)
        pred_list = torch.empty(0,len(OrgLabels)).to(self.device)
        
        for batch, data in tqdm(enumerate(self.trainloader), total=len(self.trainloader)):
            image, labels, mask = data["image"].to(self.device), data["labels"].to(self.device), data['mask'].to(self.device)
            updated_image = image.clone()
            if self.args.input_gan:
                with torch.no_grad():
                    gan_inputs = self.transform_norml(updated_image)
                    healthy_img = self.gan_pretrained.inference(gan_inputs)
                if self.args.att_module:
                    tensor_for_att = diff_map_for_att(updated_image, healthy_img, mask)
                updated_image = torch.cat((image, healthy_img), dim=1)
                
            if (epoch + 1) > self.args.n_epochs or self.args.continue_train:
                updated_image = refine_input_by_cam(self.args, self.cam_model, updated_image, mask, self.cam)
            self.cam_optimizer.zero_grad()
            outputs = self.cam_model(updated_image, tensor_for_att)
            sig_prediction = self.sigmoid(outputs)
            clf_loss = self.cam_loss(outputs, labels)
            clf_loss.backward()
            self.cam_optimizer.step()
            
            ##############################
            if self.args.segmentation < epoch + 1:
                outputs_for_seg = sig_prediction.clone()
                params_seg = {'inputs': data, 'batch_preds': outputs_for_seg, 'refined': updated_image}
                pseudo_label = get_pseudo_label(params_seg, self.cam) # (batch, w, h) pixel value in (0, nb_class - 1)
                self.seg_optimizer.zero_grad()
                seg_outputs = self.seg_model(updated_image) #(batch, nb_class, w, h), include 0 as background
                seg_loss = self.loss_seg(seg_outputs, pseudo_label.to(self.device))
                seg_loss.backward()
                self.seg_optimizer.step()
                
                total_seg_loss += seg_loss.cpu().item()
                
            ##############################            
            with torch.no_grad():
                total_cls_loss += clf_loss.cpu().item()
                batch_accuracies_metrics = calculate_classification_metrics(sig_prediction, labels)
                total_acc += Counter(batch_accuracies_metrics)
                gt_list = torch.cat((gt_list, labels))
                pred_list = torch.cat((pred_list, sig_prediction))
                
        self.cam_lr_scheduler.step()
            
        train_acc_epoch, train_loss_epoch, train_seg_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc.items()}, total_cls_loss / (batch + 1), total_seg_loss / (batch + 1)
        roc_avg, roc_class = calculate_roc(pred_list, gt_list)
        print('Epoch', str(epoch + 1), '- Train loss:', train_loss_epoch, '- Seg loss', train_seg_loss_epoch, '- Train ROC:', roc_avg,  "- ROC per class:", roc_class, "- Train acc:", train_acc_epoch)
        return train_loss_epoch, train_acc_epoch, roc_avg, roc_class

    def train(self):
        start = time.time()
        self.train_parameters()
        # Define data loaders for training and testing data
        self.trainloader = torch.utils.data.DataLoader(
                        self.dataset_train, 
                        num_workers=8,
                        batch_size=self.args.train_batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(
                        self.dataset_test,
                        num_workers=8,
                        batch_size=self.args.valid_batch_size, shuffle=False)
        best_roc = 0
        for epoch in range(0, self.num_of_epochs):
            train_loss, train_acc_matrix, train_roc_avg, train_roc_class = self.train_once(epoch)
            torch.cuda.empty_cache() ## <<<<---- AND HERE
            torch.cuda.synchronize()
            
            mark_epoch = epoch + self.args.n_epochs if self.args.continue_train else epoch
            include_valid = False
            valid_loss, valid_acc_matrxi, valid_roc_avg, valid_roc_class = None, None, None, None
            if (epoch + 1) % 5 == 0 or (epoch + 1) > self.args.n_epochs:
                valid_loss, valid_acc_matrxi, valid_roc_avg, valid_roc_class = self.valid_once(epoch)
                torch.cuda.empty_cache() ## <<<<---- AND HERE
                torch.cuda.synchronize()
                include_valid = True
                if valid_roc_avg >= best_roc:
                    save_models(self.args, epoch, self.cam_model, self.cam_optimizer, is_best=True)
                    best_roc = valid_roc_avg
            loss_dict = {
                'total_train_loss': train_loss, 
                'total_val_loss': valid_loss, 
                'total_train_acc_matrix': train_acc_matrix, 
                'total_train_roc': train_roc_avg, 
                'total_train_roc_matrix': train_roc_class,
                'total_val_acc_matrix': valid_acc_matrxi, 
                'total_val_roc': valid_roc_avg,
                'total_val_roc_matrix': valid_roc_class
            }
            save_tensorboard(self.tb, loss_dict, mark_epoch, include_valid)
            
        ## Save model for last epoch
        save_models(self.args, epoch, self.cam_model, self.cam_optimizer)
        print('final running time:', time.time() - start)
        
  
if __name__ == "__main__":
    trainer = Train()
    trainer.train()
