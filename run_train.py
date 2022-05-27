import logging
import os
import time
from collections import Counter

from pytorch_grad_cam import GradCAM
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


DEVICE_NR = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NR
logging.basicConfig(level=logging.DEBUG)

from gan_inference import load_gan_model
from dataset import (OCTDataset, train_transform, valid_transform)
from cam import save_cam_results, refine_input_by_cam, refine_input_by_background_cam, get_pseudo_label
from models import MultiTaskModel, CAM_Net, U_Net
from utils import OrgLabels, calculate_metrics, calculate_roc, save_models, save_tensorboard
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
    def __init__(self):
        self.args = Configs().parse()
        self.tb = SummaryWriter('runs/{}'.format(self.args.save_folder.split('/')[-1]))
        self.device = self.args.device 
        if self.device == "cuda":
            print("Number of GPUs: ", torch.cuda.device_count(), "Device Nbr: ", DEVICE_NR)
            
        torch.manual_seed(42)
        self.kfold = KFold(n_splits=self.args.k_folds, shuffle=False)
        self.dataset = OCTDataset(self.args, transform_train=train_transform(self.args.is_size), transform_val=valid_transform(self.args.is_size))
        self.backbone = network_class(self.args)
        self.num_class = len(OrgLabels)
        self.num_input_channel = self.dataset[0]['image'].shape[0] * 2 if self.args.input_gan else self.dataset[0]['image'].shape[0]
        self.num_of_epochs = self.args.num_iteration if self.args.continue_train else self.args.n_epochs
        self.gan_pretrained = False
        if self.args.input_gan:
            with torch.no_grad():
                path = "{}/netG.pth".format(self.args.model_gan)
                gan_pretrained_dict = torch.load(path)['state_dict'] #, map_location='cpu'
                self.gan_pretrained = load_gan_model(gan_pretrained_dict, self.device)
                print(f' Loaded Pretained GAN weights from {path}.')
        self.transform_norml = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    def train_parameters(self, fold):
        shared_model = MultiTaskModel(self.backbone, self.num_input_channel)
        cam_model = CAM_Net(shared_model, self.num_class)
        seg_model = U_Net(shared_model, self.num_class)

        if self.args.continue_train:
            print('Loading pretrained model from checkpoint {0}/fold-{1}/weights/25.pwf'.format(self.args.check_point, fold)) 
            checkpoint = torch.load('{0}/fold-{1}/weights/25.pwf'.format(self.args.check_point, fold))   
            cam_model.load_state_dict(checkpoint['state_dict'])
        
        self.cam_model = cam_model.cuda() if self.device == "cuda" else cam_model
        self.seg_model = seg_model.cuda() if self.device == "cuda" else seg_model
        
        self.cam_optimizer = optim.SGD(cam_model.parameters(), lr=self.args.lr, momentum=0.9)
        self.seg_optimizer = optim.Adam(seg_model.parameters(), lr=self.args.lr)
        
        self.loss_cam = nn.BCELoss()#nn.BCEWithLogitsLoss()#
        self.loss_seg = nn.CrossEntropyLoss()        
        
    def valid_once(self, epoch, fold):
        self.cam_model.eval()
        # Evaluationfor this fold
        total_acc_val = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
        total_loss_val = 0
        target_layers = [self.cam_model.multi_task_model.base_model[-1][-1]]
        cam = GradCAM(model=self.cam_model, use_cuda=self.device, target_layers=target_layers)
        self.testloader.dataset.set_use_train_transform(False)
        gt_list = torch.empty(0,len(OrgLabels)).to(self.device)
        pred_list = torch.empty(0,len(OrgLabels)).to(self.device)
        for batch, data in tqdm(enumerate(self.testloader), total=int(len(self.test_subsampler) / self.testloader.batch_size)):
            image, labels = data["image"].to(self.device), data["labels"].to(self.device)        
            updated_image = image.clone()
            if self.args.input_gan:
                input_for_gan = self.transform_norml(updated_image)
                gan_tensor = self.gan_pretrained.inference(input_for_gan)
                updated_image = torch.cat((image, gan_tensor), dim=1)
            if (epoch + 1) > self.args.refine_epoch_point or self.args.continue_train:
                if (epoch + 1) > (self.args.refine_epoch_point + self.args.n_refine_background) or self.args.continue_train:
                    updated_image = refine_input_by_cam(self.args, self.cam_model, updated_image, cam)
                else:
                    updated_image = refine_input_by_background_cam(self.args, self.cam_model, updated_image, cam)
            outputs = self.cam_model(updated_image)
            # maybe only for args.n_epochs in first condition
            if (epoch + 1) % 5 == 0 or self.args.continue_train or (epoch + 1) > self.args.refine_epoch_point: 
                params = {'args': self.args, 'epoch': epoch, 'model': self.cam_model, 'fold': fold, 'inputs': data, 'batch_preds': outputs, 'refined': updated_image}
                save_cam_results(params)
            with torch.no_grad():
                loss_val = self.loss_cam(outputs, labels)
                loss_dice = dsc_loss(outputs, labels)
                combined_loss = loss_dice + loss_val
                total_loss_val += combined_loss.cpu().item()
                batch_accuracies_metrics = calculate_metrics(outputs, labels)
                total_acc_val += Counter(batch_accuracies_metrics)
                gt_list = torch.cat((gt_list, labels))
                pred_list = torch.cat((pred_list, outputs))
            
        # Print accuracy
        valid_acc_epoch, valid_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc_val.items()}, total_loss_val / (batch + 1)
        roc_avg, roc_class = calculate_roc(pred_list, gt_list)
        # print(f'K-FOLD CROSS VALIDATION RESULTS FOR {fold} FOLDS')
        print('- Val loss:', valid_loss_epoch, '- Val ROC:', roc_avg,  "- ROC per class:", roc_class, "- Val acc:", valid_acc_epoch)
        return valid_loss_epoch, valid_acc_epoch, roc_avg, roc_class

    def train_once(self, epoch):
        self.cam_model.train()
        target_layers = [self.cam_model.multi_task_model.base_model[-1][-1]]
        cam = GradCAM(model=self.cam_model, use_cuda=self.device, target_layers=target_layers)
        self.trainloader.dataset.set_use_train_transform(True)
        for batch, data in tqdm(enumerate(self.trainloader), total = int(len(self.train_subsampler)/self.trainloader.batch_size)):
            image, labels = data["image"].to(self.device), data["labels"].to(self.device)
            updated_image = image.clone()
            if self.args.input_gan:
                input_for_gan = self.transform_norml(updated_image)
                gan_tensor = self.gan_pretrained.inference(input_for_gan)
                updated_image = torch.cat((image, gan_tensor), dim=1)
            if (epoch + 1) > self.args.refine_epoch_point or self.args.continue_train:
                if (epoch + 1) > (self.args.refine_epoch_point + self.args.n_refine_background) or self.args.continue_train:
                    updated_image = refine_input_by_cam(self.args, self.cam_model, updated_image, cam)
                else:
                    updated_image = refine_input_by_background_cam(self.args, self.cam_model, updated_image, cam)
            self.cam_optimizer.zero_grad()
            outputs = self.cam_model(updated_image)
            loss_train = self.loss_cam(outputs, labels)
            loss_dice = dsc_loss(outputs, labels)
            combined_loss = loss_dice + loss_train
            combined_loss.backward()
            self.cam_optimizer.step()
        
        self.cam_model.eval()
        total_acc = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
        total_loss = 0
        gt_list = torch.empty(0,len(OrgLabels)).to(self.device)
        pred_list = torch.empty(0,len(OrgLabels)).to(self.device)
        for batch, data in enumerate(self.trainloader):
            image, labels = data["image"].to(self.device), data["labels"].to(self.device)
            updated_image = image.clone()
            if self.args.input_gan:
                input_for_gan = self.transform_norml(updated_image)
                gan_tensor = self.gan_pretrained.inference(input_for_gan)
                updated_image = torch.cat((image, gan_tensor), dim=1)
            if (epoch + 1) > self.args.refine_epoch_point or self.args.continue_train:
                if (epoch + 1) > (self.args.refine_epoch_point + self.args.n_refine_background) or self.args.continue_train:
                    updated_image = refine_input_by_cam(self.args, self.cam_model, updated_image, cam)
                else:
                    updated_image = refine_input_by_background_cam(self.args, self.cam_model, updated_image, cam)
            with torch.no_grad():
                outputs = self.cam_model(updated_image)
                loss_train = self.loss_cam(outputs, labels)
                loss_dice = dsc_loss(outputs, labels)
                combined_loss = loss_dice + loss_train
                total_loss += combined_loss.cpu().item()
                batch_accuracies_metrics = calculate_metrics(outputs, labels)
                total_acc += Counter(batch_accuracies_metrics)
                gt_list = torch.cat((gt_list, labels))
                pred_list = torch.cat((pred_list, outputs))
            
        train_acc_epoch, train_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc.items()}, total_loss / (batch + 1)
        roc_avg, roc_class = calculate_roc(pred_list, gt_list)
        print('Epoch', str(epoch + 1), '- Train loss:', train_loss_epoch, '- Train ROC:', roc_avg,  "- ROC per class:", roc_class, "- Train acc:", train_acc_epoch)
        return train_loss_epoch, train_acc_epoch, roc_avg, roc_class

    def train(self):
        start = time.time()
        # K-fold Cross Validation model evaluation
        total_train_loss, total_val_loss, total_train_acc_matrix, total_val_acc_matrix = {}, {}, {}, {}
        total_val_roc_matrix, total_val_roc, total_train_roc_matrix, total_train_roc = {}, {}, {}, {}
        for fold, (train_ids, test_ids) in enumerate(self.kfold.split(self.dataset)):
            logging.info(f'---------FOLD {fold}--------')
            self.train_parameters(fold)
            # # Sample elements randomly from a given list of ids, no replacement.
            self.train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            self.test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            # Define data loaders for training and testing data in this fold
            self.trainloader = torch.utils.data.DataLoader(
                            self.dataset, 
                            num_workers=8,
                            batch_size=self.args.train_batch_size, sampler=self.train_subsampler)
            self.testloader = torch.utils.data.DataLoader(
                            self.dataset,
                            num_workers=8,
                            batch_size=self.args.valid_batch_size, sampler=self.test_subsampler, shuffle=False)
            for epoch in range(0, self.num_of_epochs):
                train_loss, train_acc_matrix, train_roc_avg, train_roc_class = self.train_once(epoch)
                mark_epoch = epoch + 25 if self.args.continue_train else epoch
                
                total_train_loss[mark_epoch] = total_train_loss[mark_epoch] + train_loss if mark_epoch in total_train_loss else 0
                total_train_acc_matrix[mark_epoch] =  {k: v + total_train_acc_matrix[mark_epoch][k] for k, v in train_acc_matrix.items()} if mark_epoch in total_train_acc_matrix else train_acc_matrix
                total_train_roc[mark_epoch] = total_train_roc[mark_epoch] + train_roc_avg if mark_epoch in total_train_roc else 0
                total_train_roc_matrix[mark_epoch] =  {k: v + total_train_roc_matrix[mark_epoch][k] for k, v in train_roc_class.items()} if mark_epoch in total_train_roc_matrix else train_roc_class
                
                if (epoch + 1) % 5 == 0 or self.args.continue_train or (epoch + 1) > self.args.refine_epoch_point:
                    valid_loss, valid_acc_matrxi, roc_avg, roc_class = self.valid_once(epoch, fold)
                    total_val_loss[mark_epoch] = total_val_loss[mark_epoch] + valid_loss if mark_epoch in total_val_loss else 0
                    total_val_acc_matrix[mark_epoch] =  {k: v + total_val_acc_matrix[mark_epoch][k] for k, v in valid_acc_matrxi.items()} if mark_epoch in total_val_acc_matrix else valid_acc_matrxi
                    total_val_roc[mark_epoch] = total_val_roc[mark_epoch] + roc_avg if mark_epoch in total_val_roc else 0
                    total_val_roc_matrix[mark_epoch] =  {k: v + total_val_roc_matrix[mark_epoch][k] for k, v in roc_class.items()} if mark_epoch in total_val_roc_matrix else roc_class
                
            ## Save model for last epoch
            save_models(self.args, fold, epoch, self.cam_model, self.cam_optimizer)
        loss_dict = {'total_train_loss': total_train_loss, 
                     'total_train_acc_matrix': total_train_acc_matrix, 
                     'total_train_roc': total_train_roc, 
                     'total_train_roc_matrix': total_train_roc_matrix,
                     'total_val_loss:': total_val_loss, 
                     'total_val_acc_matrix': total_val_acc_matrix, 
                     'total_val_roc': total_val_roc,
                     'total_val_roc_matrix': total_val_roc_matrix}
        save_tensorboard(self.tb, loss_dict, self.args.k_folds)
        print('final running time:', time.time() - start)
        


if __name__ == "__main__":
    trainer = Train()
    trainer.train()
