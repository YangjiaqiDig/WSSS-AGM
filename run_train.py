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


DEVICE_NR = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NR
logging.basicConfig(level=logging.DEBUG)

from gan_inference import load_gan_model
from dataset import (DukeDataset, OCTDataset, RESCDataset)
from refine_pseudo_label import refine_input_by_cam, get_pseudo_label
from models import MultiTaskModel, CAM_Net, U_Net
from utils import OrgLabels, save_models, save_tensorboard, save_cam_results
from metrics import calculate_classification_metrics, calculate_roc
from options import Configs


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
        self.args = Configs().parse()
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

        self.backbone = network_class(self.args)
        self.num_class = len(OrgLabels)
        self.num_input_channel = self.dataset_train[0]['image'].shape[0] * 2 if self.args.input_gan else self.dataset_train[0]['image'].shape[0]
        self.num_of_epochs = self.args.num_iteration if self.args.continue_train else self.args.n_epochs
        self.gan_pretrained = False
        if self.args.input_gan:
            with torch.no_grad():
                path = "{}/netG.pth".format(self.args.model_gan)
                gan_pretrained_dict = torch.load(path)['state_dict'] #, map_location='cpu'
                self.gan_pretrained = load_gan_model(gan_pretrained_dict, self.device)
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
        seg_model = U_Net(shared_model, self.num_class, self.args.backbone)

        if self.args.continue_train:
            print('Loading pretrained model from checkpoint {0}/weights/best.pwf'.format(self.args.check_point)) 
            checkpoint = torch.load('{0}/weights/best.pwf'.format(self.args.check_point))   
            cam_model.load_state_dict(checkpoint['state_dict'])
        
        self.cam_model = cam_model.cuda() if self.device == "cuda" else cam_model
        self.seg_model = seg_model.cuda() if self.device == "cuda" else seg_model
        
        self.cam_optimizer = optim.SGD(cam_model.parameters(), lr=self.args.lr, momentum=0.9)
        self.seg_optimizer = optim.Adam(seg_model.parameters(), lr=self.args.lr)

        self.cam_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.cam_optimizer, step_size=self.args.lr_schedule['step'], gamma=self.args.lr_schedule['gamma'])
        self.loss_cam = nn.BCELoss()#nn.BCEWithLogitsLoss()#
        self.loss_seg = nn.CrossEntropyLoss()        
        
        self.target_layers = [self.cam_model.multi_task_model.base_model[-1][-1]]
        
    def valid_once(self, epoch):
        self.cam_model.eval()
        self.seg_model.eval()
        total_acc_val = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
        total_loss_val, total_seg_loss_val = 0, 0
        cam = GradCAM(model=self.cam_model, use_cuda=self.device, target_layers=self.target_layers)
        gt_list = torch.empty(0,len(OrgLabels)).to(self.device)
        pred_list = torch.empty(0,len(OrgLabels)).to(self.device)
        for batch, data in tqdm(enumerate(self.testloader), total=len(self.testloader)):
            image, labels, mask = data["image"].to(self.device), data["labels"].to(self.device), data['mask'].to(self.device)        
            updated_image = image.clone()
            if self.args.input_gan:
                input_for_gan = self.transform_norml(updated_image)
                gan_tensor = self.gan_pretrained.inference(input_for_gan)
                updated_image = torch.cat((image, gan_tensor), dim=1)
            if self.args.mask_enhance:
                mask_clone = mask.clone()
                mask_clone[mask_clone == 0] = 0.2
                updated_image[:,:3] = updated_image[:,:3] * mask_clone
                updated_image[:,3:] = updated_image[:,3:] * mask_clone
            if (epoch + 1) > self.args.refine_epoch_point or self.args.continue_train:
                updated_image = refine_input_by_cam(self.args, self.cam_model, updated_image, mask, cam)
            outputs = self.cam_model(updated_image)
            if self.args.segmentation < epoch + 1:
                outputs_for_seg = outputs.clone()
                params_seg = {'inputs': data, 'batch_preds': outputs_for_seg, 'refined': updated_image}
                pseudo_label = get_pseudo_label(params_seg, cam) # (batch, w, h) pixel value in (0, nb_class - 1)
                seg_outputs = self.seg_model(updated_image) #(batch, nb_class, w, h), include 0 as background
                segmentation_loss = self.loss_seg(seg_outputs, pseudo_label.to(self.device))
                total_seg_loss_val += segmentation_loss.cpu().item()
            params = {'args': self.args, 'epoch': epoch, 'cam': cam, 'inputs': data, 'batch_preds': outputs, 'refined': updated_image}
            save_cam_results(params)  
                    
            with torch.no_grad():
            #     if self.args.continue_train or (epoch + 1) > self.args.refine_epoch_point:  
                combined_loss = self.cam_loss(outputs, labels)
                total_loss_val += combined_loss.cpu().item()
                batch_accuracies_metrics = calculate_classification_metrics(outputs, labels)
                total_acc_val += Counter(batch_accuracies_metrics)
                gt_list = torch.cat((gt_list, labels))
                pred_list = torch.cat((pred_list, outputs))
            
        # Print accuracy
        valid_acc_epoch, valid_loss_epoch, valid_seg_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc_val.items()}, total_loss_val / (batch + 1), total_seg_loss_val / (batch+1)
        roc_avg, roc_class = calculate_roc(pred_list, gt_list)
        print('- Val loss:', valid_loss_epoch, '- Seg loss: ', valid_seg_loss_epoch,'- Val ROC:', roc_avg,  "- ROC per class:", roc_class, "- Val acc:", valid_acc_epoch)
        return valid_loss_epoch, valid_acc_epoch, roc_avg, roc_class

    def train_once(self, epoch):
        self.cam_model.train()
        self.seg_model.train()
        cam = GradCAM(model=self.cam_model, use_cuda=self.device, target_layers=self.target_layers)

        total_acc = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
        total_loss, total_seg_loss = 0, 0
        gt_list = torch.empty(0,len(OrgLabels)).to(self.device)
        pred_list = torch.empty(0,len(OrgLabels)).to(self.device)
        
        for batch, data in tqdm(enumerate(self.trainloader), total=len(self.trainloader)):
            image, labels, mask = data["image"].to(self.device), data["labels"].to(self.device), data['mask'].to(self.device)
            updated_image = image.clone()
            if self.args.input_gan:
                input_for_gan = self.transform_norml(updated_image)
                gan_tensor = self.gan_pretrained.inference(input_for_gan)
                updated_image = torch.cat((image, gan_tensor), dim=1)
            if self.args.mask_enhance:
                mask_clone = mask.clone()
                mask_clone[mask_clone == 0] = 0.2
                updated_image[:,:3] = updated_image[:,:3] * mask_clone
                updated_image[:,3:] = updated_image[:,3:] * mask_clone
            if (epoch + 1) > self.args.refine_epoch_point or self.args.continue_train:
                updated_image = refine_input_by_cam(self.args, self.cam_model, updated_image, mask, cam)
            self.cam_optimizer.zero_grad()
            outputs = self.cam_model(updated_image)
            combined_loss = self.cam_loss(outputs, labels)
            combined_loss.backward()
            self.cam_optimizer.step()
            if self.args.segmentation < epoch + 1:
                outputs_for_seg = outputs.clone()
                params_seg = {'inputs': data, 'batch_preds': outputs_for_seg, 'refined': updated_image}
                pseudo_label = get_pseudo_label(params_seg, cam) # (batch, w, h) pixel value in (0, nb_class - 1)
                self.seg_optimizer.zero_grad()
                seg_outputs = self.seg_model(updated_image) #(batch, nb_class, w, h), include 0 as background
                segmentation_loss = self.loss_seg(seg_outputs, pseudo_label.to(self.device))
                segmentation_loss.backward()
                self.seg_optimizer.step()
                
                total_seg_loss += segmentation_loss.cpu().item()
            with torch.no_grad():
                total_loss += combined_loss.cpu().item()
                batch_accuracies_metrics = calculate_classification_metrics(outputs, labels)
                total_acc += Counter(batch_accuracies_metrics)
                gt_list = torch.cat((gt_list, labels))
                pred_list = torch.cat((pred_list, outputs))
                
        self.cam_lr_scheduler.step()
            
        train_acc_epoch, train_loss_epoch, train_seg_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc.items()}, total_loss / (batch + 1), total_seg_loss / (batch + 1)
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
            
            mark_epoch = epoch + 25 if self.args.continue_train else epoch
            include_valid = False
            valid_loss, valid_acc_matrxi, valid_roc_avg, valid_roc_class = None, None, None, None
            if (epoch + 1) % 10 == 0 or self.args.continue_train or (epoch + 1) > self.args.refine_epoch_point \
                or (epoch + 1) > self.args.segmentation:
                valid_loss, valid_acc_matrxi, valid_roc_avg, valid_roc_class = self.valid_once(epoch)
                include_valid = True
                if valid_roc_avg >= best_roc:
                    save_models(self.args, epoch, self.cam_model, self.cam_optimizer, is_best=True)
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
        
    def inference(self, infer_list=[]):
        # if not give infer list of image names, then default as infer all testing
        self.args.continue_train = True
        self.train_parameters()
        if 'our_dataset' in self.args.root_dirs:
            curr_dataset = OCTDataset(self.args, data_type='inference', infer_list=infer_list)
        elif 'RESC' in self.args.root_dirs:
            curr_dataset = RESCDataset(self.args, data_type='inference', infer_list=infer_list)
        elif 'BOE' in self.args.root_dirs:
            curr_dataset = DukeDataset(self.args, data_type='inference', infer_list=infer_list)

        infer_dataset = self.dataset_test if not len(infer_list) else curr_dataset
        dataloader = torch.utils.data.DataLoader(
            infer_dataset,
            num_workers=8,
            batch_size=self.args.valid_batch_size, shuffle=False)
        self.cam_model.eval()
        
        cam = GradCAM(model=self.cam_model, use_cuda=self.device, target_layers=self.target_layers)
        gt_list, cam_list = [], []
        for batch, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image, labels = data["image"].to(self.device), data["labels"].to(self.device)        
            updated_image = image.clone()
            if self.args.input_gan:
                input_for_gan = self.transform_norml(updated_image)
                gan_tensor = self.gan_pretrained.inference(input_for_gan)
                updated_image = torch.cat((image, gan_tensor), dim=1)
            outputs = self.cam_model(updated_image)
            # maybe only for args.n_epochs in first condition
            params = {'args': self.args, 'cam': cam, 'inputs': data, 'batch_preds': outputs, 'refined': updated_image, 'model':  self.cam_model}
            gt_res, pred_res = save_cam_results(params, is_inference=True)
            gt_list += gt_res
            cam_list += pred_res

        # import pdb; pdb.set_trace()
        from metrics import scores, record_score
        score = scores(gt_list, cam_list, n_class=3)
        print(score)
        record_score(score, 'resc', 'cls2')
        
if __name__ == "__main__":
    is_inference=False
    trainer = Train(is_inference)
    trainer.train()
    # trainer.inference()
    # trainer.inference(infer_list=['sn22698_97.bmp','sn3502_1.bmp'])
    # trainer.inference(infer_list=['DME-15307-1.jpeg',
    #                               'DME-4240465-41.jpeg', 
    #                               'DR10.jpeg',
    #                               'NORMAL-15307-1.jpeg'])
