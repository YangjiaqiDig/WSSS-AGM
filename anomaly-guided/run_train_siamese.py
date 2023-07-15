import os
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE_NR = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NR

from gan_inference import load_gan_model
from dataset import (DukeDataset, OCTDataset, RESCDataset)
from network.aam_dual import DUAL_AAM

from utils.utils_siamese import CAMGeneratorAndSave, diff_map_for_att, save_models, save_tensorboard_train, save_tensorboard_val
from utils.metrics_siamese import calculate_classification_metrics, calculate_roc, scores, record_score, visualize_metrics
from utils.loss import mIoULoss, er1_loss
from options_siamese import Configs

def network_class(args):
    if args.backbone == "resnet18":
        print("Backbone: ResNet18")
        backbone = torchvision.models.resnet18(pretrained=True)
    elif args.backbone == "vgg16":
        print("Backbone: VGG16")
        backbone = torchvision.models.vgg16(pretrained=True)
    elif args.backbone == "resnet50":
        print("Backbone: ResNet50")
        backbone = torchvision.models.resnet50(pretrained=True)
    elif args.backbone == "resnet101":
        print("Backbone: ResNet101")
        backbone = torchvision.models.resnet101(pretrained=True)
    else:
        raise NotImplementedError("No backbone found for '{}'".format(args.backbone))   
    return backbone


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
        self.backbone = network_class(self.args)
        self.label_names = Configs().get_labels() 
        self.num_class = len(self.label_names)            
        with torch.no_grad():
            path = "{}/netG.pth".format(self.args.model_gan)
            self.gan_pretrained = load_gan_model(path, DEVICE_NR)
            print(f' Loaded Pretained GAN weights from {path}.')
        self.transform_norml = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.w_cls = self.args.w_cls
        self.w_eq = self.args.w_eq
        self.w_miou = self.args.w_miou
        
        self.loss_cls_func = nn.BCEWithLogitsLoss() #no need sigmoid for this logits loss
        self.loss_miou_func = mIoULoss(n_foreground=self.num_class-1)
        self.l1_loss = nn.L1Loss()
                
    def get_total_loss(self, pred_dict, y_true, mask):
        '''classification loss'''
        # import pdb; pdb.set_trace()
        cls_con = pred_dict['cls_con']
        # cls_diff = pred_dict['cls_diff']

        loss_cls_con = self.loss_cls_func(cls_con, y_true)
        # loss_cls_diff = self.loss_cls_func(cls_diff, y_true)
        loss_cls = loss_cls_con #+ loss_cls_diff
        loss_cls *= self.w_cls

        # '''classification loss after attention'''
        # cls_att_con = pred_dict['cls_att_con']
        # cls_att_diff = pred_dict['cls_att_diff']
        # loss_cls_att_con = self.loss_cls_func(cls_att_con, y_true)
        # loss_cls_att_diff = self.loss_cls_func(cls_att_diff, y_true)
        # loss_cls_att = loss_cls_att_con + loss_cls_att_diff
        # loss_cls_att *= self.w_cls

        # '''consistency loss after attention'''
        # att_con = pred_dict['att_con']#[:,:-1,:,:]
        # att_diff = pred_dict['att_diff']#[:,:-1,:,:]
        # loss_consistancy = er1_loss(att_con, att_diff)
        # loss_consistancy *= self.w_eq
        
        '''miou loss between attention and mask'''
        # loss_miou_con = self.loss_miou_func(att_con, mask, y_true)
        # # loss_miou_diff = self.loss_miou_func(att_diff, mask, self.args.out_cam_pred_alpha)
        # loss_miou = loss_miou_con #+ loss_miou_diff
        # loss_miou *= self.w_miou

        '''reconstruction loss'''
        # real_image = orig_img[:,0].unsqueeze(1)
        # loss_recon = self.l1_loss(pred_dict['recon'], real_image) * self.args.w_recon
        return  {
                'loss_cls': loss_cls,
                # 'loss_cls_att': loss_cls_att,
                 'loss_consistancy': torch.tensor(0).to(self.device), # loss_consistancy, #torch.tensor(0).to(self.device),
                 'loss_miou': torch.tensor(0).to(self.device), # loss_miou,
                 'total': loss_cls #+ loss_miou #+ loss_consistancy  #+ loss_recon + loss_cls_att 
                 }
    
    def get_performance_metrics(self, outputs, combine_loss_dic, labels, total_acc_con, total_acc_diff, gt_list, pred_list_con, pred_list_diff, total_loss):
        sigmoid_out_con = self.sigmoid(outputs['cls_con'])
        # sigmoid_out_diff = self.sigmoid(outputs['cls_diff'])
        total_loss = {k: total_loss.get(k, 0) + combine_loss_dic.get(k, 0).cpu().item() for k in set(total_loss) & set(combine_loss_dic)}
        batch_accuracies_metrics_con = calculate_classification_metrics(sigmoid_out_con, labels)
        # batch_accuracies_metrics_diff = calculate_classification_metrics(sigmoid_out_diff, labels)
        total_acc_con += Counter(batch_accuracies_metrics_con)
        # total_acc_diff += Counter(batch_accuracies_metrics_diff)
        gt_list += labels.cpu().tolist()
        pred_list_con += sigmoid_out_con.cpu().tolist() 
        # pred_list_diff += sigmoid_out_diff.cpu().tolist()
        return total_acc_con, total_acc_diff, gt_list, pred_list_con, pred_list_diff, total_loss
                   
    def valid_once(self, epoch, testloader, multi_task_model):
        multi_task_model.eval()
        total_acc_con = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
        total_acc_diff = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
        total_loss = Counter({'loss_cls': 0, 'loss_consistancy': 0, 'loss_miou': 0, 'loss_recon': 0, 'total': 0}) # 'loss_cls_att': 0,
        gt_list, pred_list_con, pred_list_diff = [], [], []
        accum_valid_loss = []

        CAMGenerationModule = CAMGeneratorAndSave(opts=self.args, multi_task_model=multi_task_model)
        CAMGenerationModule.set_epoch(epoch)
        gt_img_list, cam_img_list = [], []
        for batch, data in tqdm(enumerate(testloader), total=len(testloader)):
            image, labels, mask = data["image"].to(self.device), data["labels"].to(self.device), data['mask'].to(self.device)      
            updated_image = image.clone()  
            with torch.no_grad():
                gan_inputs = self.transform_norml(updated_image)
                healthy_img = self.gan_pretrained.inference(gan_inputs)
                tensor_diff = diff_map_for_att(updated_image, healthy_img)
                updated_image = torch.cat((image, healthy_img, tensor_diff), dim=1)

                outputs = multi_task_model(updated_image)
                combine_loss_dic = self.get_total_loss(outputs, labels, mask)
                total_acc_con, total_acc_diff, gt_list, pred_list_con, pred_list_diff, total_loss = self.get_performance_metrics(outputs, combine_loss_dic, labels, total_acc_con, total_acc_diff, gt_list, pred_list_con, pred_list_diff, total_loss)
                
            params = {'inputs': data, 'refined': updated_image, 'outputs': outputs}
            gt_res, pred_res = CAMGenerationModule.get_cam_and_save(params)  
            gt_img_list += gt_res
            cam_img_list += pred_res                

        print(f'Num of CAM images: {len(cam_img_list)}')
        score = scores(gt_img_list, cam_img_list, n_class=self.num_class)
        print(score)
        record_score(score, self.iou_txt_path)
        
        # Print accuracy
        valid_loss_epoch =  {k: v / (batch + 1) for k, v in total_loss.items()}
        valid_loss_epoch['Epoch'] = epoch + 1
        accum_valid_loss.append(valid_loss_epoch)

        valid_acc_epoch_con, valid_acc_epoch_diff = {k: v  / (batch + 1) for k, v in total_acc_con.items()}, {k: v  / (batch + 1) for k, v in total_acc_diff.items()}
        roc_avg_con, roc_class_con = calculate_roc(pred_list_con, gt_list)
        # roc_avg_diff, roc_class_diff = calculate_roc(pred_list_diff, gt_list)
        
        acc_metrics = [{'Avg ROC': roc_avg_con, **roc_class_con, **valid_acc_epoch_con}]#, {'Avg ROC': roc_avg_diff, **roc_class_diff, **valid_acc_epoch_diff}]
        visualize_metrics(epoch, accum_valid_loss, acc_metrics)
        return valid_loss_epoch, valid_acc_epoch_con, roc_avg_con, roc_class_con, score['Mean IoU']
      

    def train_once(self, epoch, trainloader, multi_task_model, multi_optimizer, lr_scheduler):
        multi_task_model.train()
        total_acc_con = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
        total_acc_diff = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
        total_loss = Counter({'loss_cls': 0, 'loss_consistancy': 0, 'loss_miou': 0, 'loss_recon': 0, 'total': 0}) #'loss_cls_att': 0,
        gt_list, pred_list_con, pred_list_diff = [], [], []
        accum_train_loss = []
        for batch, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            image, labels, mask = data["image"].to(self.device), data["labels"].to(self.device), data['mask'].to(self.device)
            updated_image = image.clone()
            # import torchvision.utils as vutils
            # vutils.save_image(updated_image[:,0].unsqueeze(1), f'./debug/{batch}_ori.png', normalize=True, scale_each=True)
            # vutils.save_image(mask[:,0].unsqueeze(1), f'./debug/{batch}_msk.png', normalize=True, scale_each=True)
            
            ############## GAN healthy generation ################
            with torch.no_grad():
                gan_inputs = self.transform_norml(updated_image)
                healthy_img = self.gan_pretrained.inference(gan_inputs)
            # vutils.save_image(healthy_img[:,0].unsqueeze(1), f'./debug/{batch}_gan.png', normalize=True, scale_each=True)
            # vutils.save_image(healthy_v[:,0].unsqueeze(1), f'./debug/{batch}_gan_org.png', normalize=True, scale_each=True)


            ############## Activate Attention Branch ################
            tensor_diff = diff_map_for_att(updated_image, healthy_img)
            updated_image = torch.cat((image, healthy_img, tensor_diff), dim=1)
                            
            multi_optimizer.zero_grad()
            outputs = multi_task_model(updated_image)
            # import pdb; pdb.set_trace()
            combine_loss_dic = self.get_total_loss(outputs, labels, mask)
            combine_loss = combine_loss_dic['total']
            combine_loss.backward()
            multi_optimizer.step()
            lr_scheduler.step()
            
            ############## Metrics Calculation ################        
            with torch.no_grad():
                total_acc_con, total_acc_diff, gt_list, pred_list_con, pred_list_diff, total_loss = self.get_performance_metrics(outputs, combine_loss_dic, labels, total_acc_con, total_acc_diff, gt_list, pred_list_con, pred_list_diff, total_loss)
            
        train_loss_epoch =  {k: v / (batch + 1) for k, v in total_loss.items()}
        train_loss_epoch['lr'] = lr_scheduler.get_lr()[0]
        train_loss_epoch['Epoch'] = epoch + 1
        accum_train_loss.append(train_loss_epoch)

        train_acc_epoch_con, train_acc_epoch_diff = {k: v  / (batch + 1) for k, v in total_acc_con.items()}, {k: v  / (batch + 1) for k, v in total_acc_diff.items()}
        roc_avg_con, roc_class_con = calculate_roc(pred_list_con, gt_list)
        # roc_avg_diff, roc_class_diff = calculate_roc(pred_list_diff, gt_list)

        acc_metrics = [{'Avg ROC': roc_avg_con, **roc_class_con, **train_acc_epoch_con}]#, {'Avg ROC': roc_avg_diff, **roc_class_diff, **train_acc_epoch_diff}]
        visualize_metrics(epoch, accum_train_loss, acc_metrics)
        return train_loss_epoch, train_acc_epoch_con, roc_avg_con, roc_class_con

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
        if self.args.retinal_model == 'AAM':
            retinal_model_func = DUAL_AAM
        else:
            raise NotImplementedError()
        multi_task_model = retinal_model_func(self.backbone, 
                                          num_class=self.num_class,
                                          num_input_channel=self.num_input_channel, 
                                          backbone_name=self.args.backbone)
        print(multi_task_model)
        if self.args.continue_train:
            print(f"Continue training with {self.args.epochs} epochs")
            reload_epoch = self.args.ckp_epoch
            print('Loading pretrained model from checkpoint {0}/weights/{1}.pwf'.format(self.args.check_point, reload_epoch))
            checkpoint = torch.load('{0}/weights/{1}.pwf'.format(self.args.check_point, reload_epoch))   
            self.pretrained_epoch = checkpoint['epoch']
            multi_task_model.load_state_dict(checkpoint['state_dict'])
            
        multi_task_model = nn.DataParallel(multi_task_model)
        multi_task_model = multi_task_model.to(self.device)
        multi_optimizer = optim.SGD(multi_task_model.parameters(), lr=self.args.lr, momentum=0.9)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(multi_optimizer, step_size=self.args.lr_schedule['step'], gamma=self.args.lr_schedule['gamma'])

        return multi_task_model, multi_optimizer, lr_scheduler
    
    def train(self):
        start = time.time()
        dataset_train, dataset_test = self.get_dataset()
        self.num_input_channel = dataset_train[0]['image'].shape[0] * 4
        # Define data loaders for training and testing data
        
        trainloader = torch.utils.data.DataLoader(
                        dataset_train, 
                        num_workers=0,
                        batch_size=self.args.train_batch_size, shuffle=True)

        testloader = torch.utils.data.DataLoader(
                        dataset_test,
                        num_workers=0,
                        batch_size=self.args.valid_batch_size, shuffle=False)
        
        multi_task_model, multi_optimizer, lr_scheduler = self.get_models()
        best_roc = 0
        best_mIoU = 0
        epoch = 0
        for epoch in range(0, self.args.epochs):
            train_loss, train_acc_matrix, train_roc_avg, train_roc_class = self.train_once(epoch, trainloader, multi_task_model, multi_optimizer, lr_scheduler)
            # train_tensorboard_dict = {'total_train_loss': train_loss, 'total_train_acc_matrix': train_acc_matrix, 'total_train_roc': train_roc_avg, 'total_train_roc_matrix': train_roc_class,}
            # save_tensorboard_train(self.tb, train_tensorboard_dict, epoch)
            ######## Validation step per 5 epochs ##########
            if (epoch + 1) % 5 == 0:
                valid_loss, valid_acc_matrxi, valid_roc_avg, valid_roc_class, mIoU = self.valid_once(epoch, testloader, multi_task_model)
            #     if valid_roc_avg >= best_roc:
            #         save_models(self.args, epoch, multi_task_model, multi_optimizer, best_type='cls')
            #         best_roc = valid_roc_avg
            #     if mIoU > best_mIoU:
            #         save_models(self.args, epoch, multi_task_model, multi_optimizer, best_type='pseudo')
            #         best_mIoU = mIoU
            #     # eval_tensorboard_dict = {'total_val_loss': valid_loss, 'total_val_acc_matrix': valid_acc_matrxi, 'total_val_roc': valid_roc_avg, 'total_val_roc_matrix': valid_roc_class}
            #     # save_tensorboard_val(self.tb, eval_tensorboard_dict, epoch)
            # save_models(self.args, epoch, multi_task_model, multi_optimizer)
        
        print('final running time:', time.time() - start)
        
  
if __name__ == "__main__":
    trainer = Train()
    trainer.train()
