import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE_NR = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NR

import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from network.segment import UNet, SegFormerModel

from dataset_seg import DukeDataset, OCTDataset, RESCDatasetSeg

from utils.utils_full_seg import (
    LossFunctions,
    OrgLabels,
    get_num_classes,
    restore_and_save,
    save_models,
    save_tensorboard_train,
    save_tensorboard_val,
)
from options_seg import Configs
from utils.metrics_full_seg import Dice, scores


def dsc_loss(y_pred, y_true, varepsilon=1.0e-8):
    epsilon = 1.0e-8
    y_true = y_true.float()
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

    numerator = 2 * (y_true * y_pred * (1 - y_pred)).sum() + varepsilon
    denominator = (y_true + y_pred * (1 - y_pred)).sum() + varepsilon

    return 1 - numerator / denominator


class Train:
    def __init__(self, is_inference=False):
        self.args = Configs().parse(is_inference)
        self.iou_txt_path = self.args.save_folder.split("/")[1]
        if not is_inference:
            self.tb = SummaryWriter("runs/{}".format(self.args.save_folder[8:]))
        self.device = self.args.device
        if self.device == "cuda":
            print(
                "Number of GPUs: ", torch.cuda.device_count(), "Device Nbr: ", DEVICE_NR
            )

        self.num_class = len(OrgLabels)
        self.loss_fn = LossFunctions()
        self.lesion_in_layer = {0: [], 1: [5, 6, 7, 8], 2: [9, 10]} # PED: RPE and Choroid,

        # if "layer" in self.args.loss_type:
        #     self.model_G = UNet_4mp(n_channels=1, n_classes=12)
        #     self.model_G = nn.DataParallel(self.model_G).cuda()
        #     ckpt_path = "/scr2/xhu/jiaqi/gan_and_str/P_Net_Anomaly_Detection/str_models/oct_str_model.pth.tar"
        #     if os.path.isfile(ckpt_path):
        #         print("=> loading checkpoint '{}'".format(ckpt_path))
        #         checkpoint = torch.load(ckpt_path)
        #         self.model_G.load_state_dict(checkpoint["state_dict_G"])
        #         print(
        #             "=> loaded checkpoint '{}' (epoch {})".format(
        #                 ckpt_path, checkpoint["epoch"]
        #             )
        #         )
        #     self.model_G.eval()

    def valid_once(self, epoch, testloader, multi_task_model):
        multi_task_model.eval()
        total_seg_loss_val = 0

        gt_img_list, cam_img_list = [], []
        for batch, data in tqdm(enumerate(testloader), total=len(testloader)):
            image, labels, layer_mask = data["image"], data["labels"].to(self.device), data["layer_mask"].to(self.device)
            # import pdb; pdb.set_trace()
            input_tensor = image.to(self.device)
            # input_tensor = self.feature_extractor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                seg_out = multi_task_model(input_tensor)

            # layer probablity map
            # layer_mask = None
            # if "layer" in self.args.loss_type:
            #     img_for_layer = image.clone()[:, :1]
            #     img_for_layer[img_for_layer > 0.95] = 0
            #     with torch.no_grad():
            #         layer_mask = self.model_G(img_for_layer)

            gt_res, pred_res = restore_and_save(
                data["path"],
                seg_out,
                labels,
                epoch,
                opts=self.args,
                save_results=self.args.save_results,
            )
            gt_img_list += gt_res
            cam_img_list += pred_res

            with torch.no_grad():
                loss = self.loss_fn.get_loss(
                    labels,
                    seg_out,
                    self.args.loss_type,
                    layer_pred=layer_mask,
                    lesion_in_layer=self.lesion_in_layer,
                    scale_factor=self.args.layer_scale,
                )
                total_seg_loss_val += loss.cpu().item()

        score = scores(gt_img_list, cam_img_list, n_class=get_num_classes() + 1)

        valid_mean_acc = score["Mean Accuracy"]
        valid_mean_miou = score["Mean IoU"]
        valid_class_miou = score["Class IoU"]
        # Print accuracy
        valid_seg_loss_epoch = total_seg_loss_val / (batch + 1)
        print(
            "- Seg loss: ",
            valid_seg_loss_epoch,
            "- Val acc:",
            valid_mean_acc,
            "- Val mIoU:",
            valid_mean_miou,
            "- Val class mIoU:",
            valid_class_miou,
        )
        return valid_seg_loss_epoch, valid_mean_acc, valid_mean_miou, valid_class_miou

    def train_once(
        self, epoch, trainloader, multi_task_model, multi_optimizer, lr_scheduler
    ):
        multi_task_model.train()
        total_seg_loss = 0
        gt_img_list, pred_img_list = [], []
        for batch, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            image, labels, layer_mask = data["image"], data["labels"].to(self.device), data["layer_mask"].to(self.device)
            # import pdb; pdb.set_trace()
            # layer probablity map
            # layer_mask = None
            # if "layer" in self.args.loss_type:
                # img_for_layer = image.clone()[:, :1]
                # img_for_layer[img_for_layer > 0.95] = 0
                # with torch.no_grad():
                #     layer_mask = self.model_G(img_for_layer)
                # import pdb; pdb.set_trace()
            input_tensor = image.to(self.device)

            multi_optimizer.zero_grad()
            seg_out = multi_task_model(input_tensor)

            combine_loss = self.loss_fn.get_loss(
                labels,
                seg_out,
                self.args.loss_type,
                layer_pred=layer_mask,
                lesion_in_layer=self.lesion_in_layer,
                scale_factor=self.args.layer_scale,
            )
            combine_loss.backward()
            multi_optimizer.step()

            ############## Metrics Calculation ################
            with torch.no_grad():
                total_seg_loss += combine_loss.cpu().item()
                gt_img_list.append(labels.cpu().numpy())
                pred_img_list.append(torch.argmax(seg_out, dim=1).cpu().numpy())
        lr_scheduler.step()
        train_seg_loss_epoch = total_seg_loss / (batch + 1)
        score = scores(
            np.vstack(gt_img_list),
            np.vstack(pred_img_list),
            n_class=get_num_classes() + 1,
        )
        train_mean_acc = score["Mean Accuracy"]
        train_mean_miou = score["Mean IoU"]
        train_class_miou = score["Class IoU"]
        print(
            "Epoch",
            str(epoch + 1),
            "- Seg loss",
            train_seg_loss_epoch,
            "- Train acc:",
            train_mean_acc,
            "- Train mIoU:",
            train_mean_miou,
            "- Train class mIoU:",
            train_class_miou,
        )
        return total_seg_loss, train_mean_acc, train_mean_miou, train_class_miou

    def get_dataset(self):
        if "our_dataset" in self.args.root_dirs:
            dataset_train = OCTDataset(self.args, data_type="train")
            dataset_test = OCTDataset(self.args, data_type="test")
        elif "RESC" in self.args.root_dirs:
            dataset_train = RESCDatasetSeg(self.args, data_type="train")
            dataset_test = RESCDatasetSeg(self.args, data_type="test")
        else:
            dataset_train = DukeDataset(self.args, data_type="train")
            dataset_test = DukeDataset(self.args, data_type="test")

        return dataset_train, dataset_test

    def get_models(self):
        if self.args.retinal_model == "UNet":
            multi_task_model = UNet(n_classes=self.num_class)
        elif self.args.retinal_model == "Segformer":
            multi_task_model = SegFormerModel(n_classes=self.num_class)
        # print(multi_task_model)
        if self.args.continue_train:
            reload_epoch = 40 if self.args.ckp_epoch == "last" else self.args.ckp_epoch
            print(
                "Loading pretrained model from checkpoint {0}/weights/{1}.pwf".format(
                    self.args.check_point, reload_epoch
                )
            )
            checkpoint = torch.load(
                "{0}/weights/{1}.pwf".format(self.args.check_point, reload_epoch)
            )
            self.pretrained_epoch = checkpoint["epoch"]
            multi_task_model.load_state_dict(checkpoint["state_dict"])

        # multi_task_model = nn.DataParallel(multi_task_model)
        multi_task_model = multi_task_model.to(self.device)
        # multi_optimizer = optim.SGD(
        #     multi_task_model.parameters(), lr=self.args.lr, momentum=0.9
        # ) # SDG
        multi_optimizer = optim.Adam(multi_task_model.parameters(), lr=self.args.lr)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            multi_optimizer,
            step_size=self.args.lr_schedule["step"],
            gamma=self.args.lr_schedule["gamma"],
        )

        return multi_task_model, multi_optimizer, lr_scheduler

    def train(self):
        start = time.time()
        dataset_train, dataset_test = self.get_dataset()
        # Define data loaders for training and testing data

        trainloader = torch.utils.data.DataLoader(
            dataset_train,
            num_workers=8,
            batch_size=self.args.train_batch_size,
            shuffle=True,
        )

        testloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=8,
            batch_size=self.args.valid_batch_size,
            shuffle=False,
        )

        multi_task_model, multi_optimizer, lr_scheduler = self.get_models()
        best_mIoU = 0

        # Regular training process
        epoch = 0 if not self.args.continue_train else self.pretrained_epoch + 1
        for epoch in range(epoch, epoch + self.args.epochs):
            (
                total_seg_loss,
                train_mean_acc,
                train_mean_miou,
                train_class_miou,
            ) = self.train_once(
                epoch, trainloader, multi_task_model, multi_optimizer, lr_scheduler
            )
            # train_tensorboard_dict = {'total_train_loss': train_loss, 'total_train_acc_matrix': train_acc_matrix, 'total_train_roc': train_roc_avg, 'total_train_roc_matrix': train_roc_class,}
            # save_tensorboard_train(self.tb, train_tensorboard_dict, epoch)
            ######## Validation step per 5 epochs ##########
            if (epoch + 1) % 5 == 0:
                (
                    valid_seg_loss_epoch,
                    valid_mean_acc,
                    valid_mean_miou,
                    valid_class_miou,
                ) = self.valid_once(epoch, testloader, multi_task_model)
                if valid_mean_miou > best_mIoU:
                    save_models(
                        self.args,
                        epoch,
                        multi_task_model,
                        multi_optimizer,
                        best_type="pseudo",
                    )
                    best_mIoU = valid_mean_miou
                # eval_tensorboard_dict = {'total_val_loss': valid_loss, 'total_val_acc_matrix': valid_acc_matrxi, 'total_val_roc': valid_roc_avg, 'total_val_roc_matrix': valid_roc_class}
                # save_tensorboard_val(self.tb, eval_tensorboard_dict, epoch)
            if epoch == self.args.epochs - 1:
                ## Save model for last regular train epoch
                save_models(self.args, epoch, multi_task_model, multi_optimizer)

        print("final running time:", time.time() - start)


if __name__ == "__main__":
    trainer = Train()
    trainer.train()