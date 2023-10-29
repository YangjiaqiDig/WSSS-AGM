import os
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from utils.loss import dsc_loss
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE_NR = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NR

from dataset import DukeDataset, OCTDataset, RESCDatasetSeg
from network.unet_model import UNetWeakly, UNetWeaklyBranch1, UNetWeaklyBranch2, UNetWeaklyBranch2Binary

from utils.utils import (
    OrgLabels,
    get_cam_and_save,
    normalize_cam,
    post_process_cam,
    save_models,
    save_tensorboard_train,
    save_tensorboard_val,
)
from utils.metrics import calculate_classification_metrics, calculate_roc, scores, record_score
from options import Configs
import numpy as np

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

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.num_class = len(OrgLabels)
        self.loss_cls_bce = (
            nn.BCEWithLogitsLoss()
        )  # no need sigmoid for this logits loss
        self.loss_layer_ce = nn.CrossEntropyLoss()

    def get_cls_loss(self, orig_pred, layer_pred, seg_cls_pred, orig_gt, layer_gt):
        if orig_pred is not None:
            loss_orig = self.loss_cls_bce(orig_pred, orig_gt)
        else:
            # create a dummy loss tensor 0 in cuda
            loss_orig = torch.tensor(0).to(self.device)
        if layer_pred is not None:
            loss_layer = self.loss_layer_ce(layer_pred, layer_gt)
        else:
            # create a dummy loss tensor 0 in cuda
            loss_layer = torch.tensor(0).to(self.device)
        if seg_cls_pred is not None:
            loss_seg = self.loss_cls_bce(seg_cls_pred, orig_gt)
        else:
            # create a dummy loss tensor 0 in cuda
            loss_seg = torch.tensor(0).to(self.device)
        # import pdb; pdb.set_trace()
        return loss_orig + loss_layer + loss_seg

    def valid_once(self, epoch, testloader, multi_task_model):
        multi_task_model.eval()
        total_acc_val = Counter({"acc": 0, "f1m": 0, "f1mi": 0})
        total_cls_loss_val = 0
        gt_list = []
        pred_list = []

        gt_img_list, seg_img_list, cam_img_list = [], [], []
        for batch, data in tqdm(enumerate(testloader), total=len(testloader)):
            image, labels, binary_labels, layer_img = (
                data["image"].to(self.device),
                data["labels"].to(self.device),
                data["binary_labels"].to(self.device),
                data["layer_img"].to(self.device),
            )
        
            with torch.no_grad():
                orig_cls_res, layer_cls_res, seg_res, cams, seg_class_pred = multi_task_model([image, layer_img])
                combine_loss = self.get_cls_loss(orig_cls_res, layer_cls_res, seg_class_pred, labels, binary_labels)

                if orig_cls_res is None and layer_cls_res is not None:
                    cams_with_bg_score, pred_cam_labels = post_process_cam(cams["final_cam"], F.one_hot(binary_labels, num_classes=2), bg_score=self.args.out_cam_pred_alpha)
                else:
                    cams_with_bg_score, pred_cam_labels = post_process_cam(cams["final_cam"], labels, bg_score=self.args.out_cam_pred_alpha)
                if self.args.seg_start_epoch <= epoch:
                    seg_loss =  self.loss_layer_ce(seg_res, pred_cam_labels.to(self.device))
                    combine_loss += seg_loss

                orig_cam = normalize_cam(cams["orig_cam"])
                layer_cam = normalize_cam(cams["layer_cam"])
                gt_res, ready_seg, ready_cam_pred = get_cam_and_save(seg_res, cams_with_bg_score, data, self.args, epoch, orig_cam, layer_cam)
                gt_img_list += gt_res
                seg_img_list += ready_seg
                cam_img_list += ready_cam_pred

                total_cls_loss_val += combine_loss.cpu().item()
                if orig_cls_res is None and layer_cls_res is not None:
                    layer_binary_prediction = self.softmax(layer_cls_res)
                    batch_accuracies_metrics = calculate_classification_metrics(
                        layer_binary_prediction, F.one_hot(binary_labels, num_classes=2)
                    )
                    gt_list += F.one_hot(binary_labels, num_classes=2).cpu().tolist()
                    pred_list += layer_binary_prediction.cpu().tolist()
                else:
                    sig_prediction = self.sigmoid(orig_cls_res)
                    batch_accuracies_metrics = calculate_classification_metrics(
                        sig_prediction, labels
                    )
                    gt_list += labels.cpu().tolist()
                    pred_list += sig_prediction.cpu().tolist()
                total_acc_val += Counter(batch_accuracies_metrics)

        print(f"Num of CAM images: {len(pred_list)}")

        score_cam = scores(gt_img_list, cam_img_list, n_class=self.num_class)
        score_comb = score_cam
        print("CAM:\n", score_cam)
        if self.args.seg_start_epoch <= epoch:
            score_seg = scores(gt_img_list, seg_img_list, n_class=self.num_class)
            print("Seg (target):\n", score_seg)
            score_comb = score_seg
        

        # Print accuracy
        valid_acc_epoch, valid_loss_epoch = {
            k: v / (batch + 1) for k, v in total_acc_val.items()
        }, total_cls_loss_val / (batch + 1)
        roc_avg, roc_class = calculate_roc(pred_list, gt_list)
        print(
            "- Val loss:",
            valid_loss_epoch,
            "- Val ROC:",
            roc_avg,
            "- ROC per class:",
            roc_class,
            "- Val acc:",
            valid_acc_epoch,
        )
        return valid_loss_epoch, valid_acc_epoch, roc_avg, roc_class, score_comb["Mean IoU"]

    def train_once(
        self, epoch, trainloader, multi_task_model, multi_optimizer, lr_scheduler
    ):
        multi_task_model.train()

        total_acc = Counter({"acc": 0, "f1m": 0, "f1mi": 0})
        total_cls_loss = 0
        gt_list, pred_list = [], []
        for batch, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            image, labels, binary_labels, layer_img = (
                data["image"].to(self.device),
                data["labels"].to(self.device),
                data["binary_labels"].to(self.device),
                data["layer_img"].to(self.device),
            )
            multi_optimizer.zero_grad()
            orig_cls_res, layer_cls_res, seg_res, cams, seg_class_pred = multi_task_model([image, layer_img])

            combine_loss = self.get_cls_loss(orig_cls_res, layer_cls_res, seg_class_pred, labels, binary_labels)
            if self.args.seg_start_epoch <= epoch:
                if orig_cls_res is None and layer_cls_res is not None:
                    _, pred_cam_labels = post_process_cam(cams["final_cam"], F.one_hot(binary_labels, num_classes=2), bg_score=self.args.out_cam_pred_alpha)
                _, pred_cam_labels = post_process_cam(cams["final_cam"], labels, bg_score=self.args.out_cam_pred_alpha)
                # import pdb; pdb.set_trace()
                seg_loss =  self.loss_layer_ce(seg_res, pred_cam_labels.to(self.device))
                combine_loss += seg_loss
            combine_loss.backward()
            multi_optimizer.step()

            ############## Metrics Calculation ################
            with torch.no_grad():
                total_cls_loss += combine_loss.cpu().item()
                if orig_cls_res is None and layer_cls_res is not None:
                    layer_binary_prediction = self.softmax(layer_cls_res)
                    batch_accuracies_metrics = calculate_classification_metrics(
                        layer_binary_prediction, F.one_hot(binary_labels, num_classes=2)
                    )
                    gt_list += F.one_hot(binary_labels, num_classes=2).cpu().tolist()
                    pred_list += layer_binary_prediction.cpu().tolist()
                else:
                    sig_prediction = self.sigmoid(orig_cls_res)
                    batch_accuracies_metrics = calculate_classification_metrics(
                        sig_prediction, labels
                    )
                    gt_list += labels.cpu().tolist()
                    pred_list += sig_prediction.cpu().tolist()

                total_acc += Counter(batch_accuracies_metrics)
        lr_scheduler.step()
        train_acc_epoch, train_loss_epoch = {
            k: v / (batch + 1) for k, v in total_acc.items()
        }, total_cls_loss / (batch + 1)
        roc_avg, roc_class = calculate_roc(pred_list, gt_list)
        print(
            "Epoch",
            str(epoch + 1),
            "- Train loss:",
            train_loss_epoch,
            "- Train ROC:",
            roc_avg,
            "- ROC per class:",
            roc_class,
            "- Train acc:",
            train_acc_epoch,
        )
        return train_loss_epoch, train_acc_epoch, roc_avg, roc_class

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
        multi_task_model = UNetWeakly(n_classes=self.num_class)
        print(multi_task_model)
        if self.args.continue_train:
            print(
                f"Continue training with regular {self.args.reg_epochs} epochs, {self.args.iter_epochs} refinement epochs"
            )
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
        best_roc = 0
        best_mIoU = 0

        epoch = 0
        for epoch in range(0, self.args.n_epochs):
            (
                train_loss,
                train_acc_matrix,
                train_roc_avg,
                train_roc_class,
            ) = self.train_once(
                epoch, trainloader, multi_task_model, multi_optimizer, lr_scheduler
            )
            train_tensorboard_dict = {
                "total_train_loss": train_loss,
                "total_train_acc_matrix": train_acc_matrix,
                "total_train_roc": train_roc_avg,
                "total_train_roc_matrix": train_roc_class,
            }
            save_tensorboard_train(self.tb, train_tensorboard_dict, epoch)
            ######## Validation step per 5 epochs ##########
            if (epoch + 1) % 5 == 0:
                (
                    valid_loss,
                    valid_acc_matrxi,
                    valid_roc_avg,
                    valid_roc_class,
                    mIoU,
                ) = self.valid_once(epoch, testloader, multi_task_model)
                if valid_roc_avg >= best_roc:
                    save_models(
                        self.args,
                        epoch,
                        multi_task_model,
                        multi_optimizer,
                        best_type="cls",
                    )
                    best_roc = valid_roc_avg
                if mIoU > best_mIoU:
                    save_models(
                        self.args,
                        epoch,
                        multi_task_model,
                        multi_optimizer,
                        best_type="pseudo",
                    )
                    best_mIoU = mIoU
                eval_tensorboard_dict = {
                    "total_val_loss": valid_loss,
                    "total_val_acc_matrix": valid_acc_matrxi,
                    "total_val_roc": valid_roc_avg,
                    "total_val_roc_matrix": valid_roc_class,
                }
                save_tensorboard_val(self.tb, eval_tensorboard_dict, epoch)

        save_models(self.args, epoch, multi_task_model, multi_optimizer)

        print("final running time:", time.time() - start)


if __name__ == "__main__":
    trainer = Train()
    trainer.train()
