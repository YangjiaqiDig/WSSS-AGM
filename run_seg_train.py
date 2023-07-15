import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE_NR = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NR

import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from network.segment import SegFormerModel, UNetWeakly


from dataset_seg import (DukeDataset, OCTDataset, RESCDataset)

from utils.utils_full_seg import OrgLabels, get_num_classes, save_models, save_tensorboard_train, save_tensorboard_val, get_seg_and_save
from options_seg import Configs
from utils.metrics_full_seg import scores, record_score, calculate_classification_metrics, calculate_roc
from transformers import SegformerFeatureExtractor


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
        self.iou_txt_path = self.args.save_folder.split('/')[1]
        if not is_inference:
            self.tb = SummaryWriter('runs/{}'.format(self.args.save_folder[8:]))
        self.device = self.args.device 
        if self.device == "cuda":
            print("Number of GPUs: ", torch.cuda.device_count(), "Device Nbr: ", DEVICE_NR)
            
        self.sigmoid = nn.Sigmoid()
        self.num_class = len(OrgLabels)            
        self.transform_norml = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.w_ce = self.args.w_ce
        self.w_dice = self.args.w_dice
        
        self.loss_cls_func = nn.BCEWithLogitsLoss() #no need sigmoid for this logits loss
        self.loss_seg_func = nn.CrossEntropyLoss() 
        model_name = "nvidia/segformer-b5-finetuned-ade-640-640"
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)

        
    def get_cls_loss(self, y_pred, y_true):
        loss_ce = self.loss_cls_func(y_pred, y_true)
        loss_dice = dsc_loss(y_pred, y_true)
        return  self.w_ce * loss_ce + self.w_dice * loss_dice  
                   
    def valid_once(self, epoch, testloader, multi_task_model):
        multi_task_model.eval()
        total_acc_val = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
        total_cls_loss_val, total_seg_loss_val = 0, 0
        gt_list = []
        pred_list = []

        gt_img_list, cam_img_list = [], []
        for batch, data in tqdm(enumerate(testloader), total=len(testloader)):
            image, labels, mask = data["image"], data["labels"].to(self.device), data['mask'].to(self.device)   
            # import pdb; pdb.set_trace()     
            input_tensor = image.to(self.device)
            # input_tensor = self.feature_extractor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                cls_outputs, seg_out = multi_task_model(input_tensor)
                sig_prediction = self.sigmoid(cls_outputs)
                
            gt_res, pred_res = get_seg_and_save(data["path"], seg_out, labels, epoch, opts=self.args, save_results=self.args.save_results)  
            gt_img_list += gt_res
            cam_img_list += pred_res
            
            with torch.no_grad():
                clf_loss = self.get_cls_loss(cls_outputs, labels)
                total_cls_loss_val += clf_loss.cpu().item()
                batch_accuracies_metrics = calculate_classification_metrics(sig_prediction, labels)
                total_acc_val += Counter(batch_accuracies_metrics)
                gt_list += labels.cpu().tolist()
                pred_list += sig_prediction.tolist()
        
        score = scores(gt_img_list, cam_img_list, n_class=get_num_classes()+1)
        print(score)
        record_score(score, self.iou_txt_path)
        
        # Print accuracy
        valid_acc_epoch, valid_loss_epoch, valid_seg_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc_val.items()}, total_cls_loss_val / (batch + 1), total_seg_loss_val / (batch+1)
        roc_avg, roc_class = calculate_roc(pred_list, gt_list)
        print('- Val loss:', valid_loss_epoch, '- Seg loss: ', valid_seg_loss_epoch,'- Val ROC:', roc_avg,  "- ROC per class:", roc_class, "- Val acc:", valid_acc_epoch)
        return valid_loss_epoch, valid_acc_epoch, roc_avg, roc_class, score['Mean IoU']
      

    def train_once(self, epoch, trainloader, multi_task_model, multi_optimizer, lr_scheduler):
        multi_task_model.train()        
        total_acc = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
        total_cls_loss, total_seg_loss = 0, 0
        gt_list, pred_list = [], []
        for batch, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            image, labels, mask = data["image"], data["labels"].to(self.device), data['mask'].to(self.device)
            input_tensor = image.to(self.device)
            # input_tensor = self.feature_extractor(image, return_tensors="pt").to(self.device)

            multi_optimizer.zero_grad()
            cls_outputs, seg_out = multi_task_model(input_tensor)
            sig_prediction = self.sigmoid(cls_outputs)
            
            combine_loss = self.get_cls_loss(cls_outputs, labels)
            combine_loss.backward()
            multi_optimizer.step()
            
            ############## Metrics Calculation ################            
            with torch.no_grad():
                total_cls_loss += combine_loss.cpu().item()
                batch_accuracies_metrics = calculate_classification_metrics(sig_prediction, labels)
                total_acc += Counter(batch_accuracies_metrics)
                gt_list += labels.cpu().tolist()
                pred_list += sig_prediction.cpu().tolist() 
        lr_scheduler.step()
        train_acc_epoch, train_loss_epoch, train_seg_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc.items()}, total_cls_loss / (batch + 1), total_seg_loss / (batch + 1)
        roc_avg, roc_class = calculate_roc(pred_list, gt_list)
        print('Epoch', str(epoch + 1), '- Train loss:', train_loss_epoch, '- Seg loss', train_seg_loss_epoch, '- Train ROC:', roc_avg,  "- ROC per class:", roc_class, "- Train acc:", train_acc_epoch)
        return train_loss_epoch, train_acc_epoch, roc_avg, roc_class

    def get_dataset(self):
        if 'our_dataset' in self.args.root_dirs:
            dataset_train = OCTDataset(self.args, data_type='train')
            dataset_test = OCTDataset(self.args, data_type='test')
        elif 'RESC' in self.args.root_dirs:
            dataset_train = RESCDataset(self.args, data_type='train')
            dataset_test = RESCDataset(self.args, data_type='test')
        else:
            dataset_train = DukeDataset(self.args, data_type='train')
            dataset_test = DukeDataset(self.args, data_type='test')
        
        return dataset_train, dataset_test
    
    def get_models(self):
        multi_task_model = SegFormerModel(n_classes=self.num_class)
        # multi_task_model = UNetWeakly(n_classes=self.num_class)
        # print(multi_task_model)
        if self.args.continue_train:
            reload_epoch = 40 if self.args.ckp_epoch == 'last' else self.args.ckp_epoch
            print('Loading pretrained model from checkpoint {0}/weights/{1}.pwf'.format(self.args.check_point, reload_epoch))
            checkpoint = torch.load('{0}/weights/{1}.pwf'.format(self.args.check_point, reload_epoch))   
            self.pretrained_epoch = checkpoint['epoch']
            multi_task_model.load_state_dict(checkpoint['state_dict'])
            
        # multi_task_model = nn.DataParallel(multi_task_model)
        multi_task_model = multi_task_model.to(self.device)
        multi_optimizer = optim.SGD(multi_task_model.parameters(), lr=self.args.lr, momentum=0.9)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(multi_optimizer, step_size=self.args.lr_schedule['step'], gamma=self.args.lr_schedule['gamma'])

        return multi_task_model, multi_optimizer, lr_scheduler
    
    def train(self):
        start = time.time()
        dataset_train, dataset_test = self.get_dataset()
        # Define data loaders for training and testing data
        
        trainloader = torch.utils.data.DataLoader(
                        dataset_train, 
                        num_workers=8,
                        batch_size=self.args.train_batch_size, shuffle=True)

        testloader = torch.utils.data.DataLoader(
                        dataset_test,
                        num_workers=8,
                        batch_size=self.args.valid_batch_size, shuffle=False)
        
        multi_task_model, multi_optimizer, lr_scheduler = self.get_models()
        best_roc = 0
        best_mIoU = 0
        # TODO: the regular training epoch must consider the continue train as well...
        # Regular training process
        epoch = 0
        for epoch in range(0, self.args.reg_epochs):
            train_loss, train_acc_matrix, train_roc_avg, train_roc_class = self.train_once(epoch, trainloader, multi_task_model, multi_optimizer, lr_scheduler)
            train_tensorboard_dict = {'total_train_loss': train_loss, 'total_train_acc_matrix': train_acc_matrix, 'total_train_roc': train_roc_avg, 'total_train_roc_matrix': train_roc_class,}
            save_tensorboard_train(self.tb, train_tensorboard_dict, epoch)
            ######## Validation step per 5 epochs ##########
            if (epoch + 1) % 5 == 0:
                valid_loss, valid_acc_matrxi, valid_roc_avg, valid_roc_class, mIoU = self.valid_once(epoch, testloader, multi_task_model)
                if valid_roc_avg >= best_roc:
                    save_models(self.args, epoch, multi_task_model, multi_optimizer, best_type='cls')
                    best_roc = valid_roc_avg
                if mIoU > best_mIoU:
                    save_models(self.args, epoch, multi_task_model, multi_optimizer, best_type='pseudo')
                    best_mIoU = mIoU
                eval_tensorboard_dict = {'total_val_loss': valid_loss, 'total_val_acc_matrix': valid_acc_matrxi, 'total_val_roc': valid_roc_avg, 'total_val_roc_matrix': valid_roc_class}
                save_tensorboard_val(self.tb, eval_tensorboard_dict, epoch)
            if epoch == self.args.reg_epochs - 1:
                ## Save model for last regular train epoch
                save_models(self.args, epoch, multi_task_model, multi_optimizer)
        
        print('final running time:', time.time() - start)
        
  
if __name__ == "__main__":
    trainer = Train()
    trainer.train()
