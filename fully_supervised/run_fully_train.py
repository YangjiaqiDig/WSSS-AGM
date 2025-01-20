import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE_NR = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NR

import time

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from network.unet_model import UNet
from network.segformer_model import SegFormerModel

from dataset_seg import RESCDatasetSeg, DukeDatasetSeg

from utils.utils_full_seg import (
    LossFunctions,
    OrgLabels,
    get_num_classes,
    restore_and_save,
    save_models,
)
from options_seg import Configs
from utils.metrics_full_seg import scores


class Train:
    def __init__(self, is_inference=False):
        self.args = Configs().parse(is_inference)
        self.device = self.args.device
        if self.device == "cuda":
            print(
                "Number of GPUs: ", torch.cuda.device_count(), "Device Nbr: ", DEVICE_NR
            )

        self.num_class = len(OrgLabels)
        self.loss_fn = LossFunctions()
        self.lesion_in_layer = {
            0: [],
            1: [5, 6, 7, 8],
            2: [9, 10],
        }  # PED: RPE and Choroid,

    def valid_once(self, epoch, testloader, multi_task_model):
        multi_task_model.eval()
        total_seg_loss_val = 0

        gt_img_list, cam_img_list = [], []
        for batch, data in tqdm(enumerate(testloader), total=len(testloader)):
            image, labels = (
                data["image"],
                data["labels"].to(self.device),
            )
            input_tensor = image.to(self.device)
            with torch.no_grad():
                seg_out = multi_task_model(input_tensor)

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
            image, labels = (
                data["image"],
                data["labels"].to(self.device),
            )
            input_tensor = image.to(self.device)

            multi_optimizer.zero_grad()
            seg_out = multi_task_model(input_tensor)

            combine_loss = self.loss_fn.get_loss(
                labels,
                seg_out,
                self.args.loss_type,
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
        if "RESC" in self.args.root_dirs:
            dataset_train = RESCDatasetSeg(self.args, data_type="train")
            dataset_test = RESCDatasetSeg(self.args, data_type="test")
        elif "2015_BOE_Chiu" in self.args.root_dirs:
            dataset_train = DukeDatasetSeg(self.args, data_type="train")
            dataset_test = DukeDatasetSeg(self.args, data_type="test")
        return dataset_train, dataset_test

    def get_models(self):
        if self.args.retinal_model == "UNet":
            multi_task_model = UNet(n_classes=self.num_class)
        elif self.args.retinal_model == "Segformer":
            multi_task_model = SegFormerModel(n_classes=self.num_class, version="b5")

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

        multi_task_model = multi_task_model.to(self.device)
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
            if epoch == self.args.epochs - 1:
                ## Save model for last regular train epoch
                save_models(self.args, epoch, multi_task_model, multi_optimizer)

        print("final running time:", time.time() - start)


if __name__ == "__main__":
    trainer = Train()
    trainer.train()