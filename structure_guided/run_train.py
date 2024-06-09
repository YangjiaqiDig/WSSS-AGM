import os
import time
import pickle
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE_NR = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NR

from transformers import (
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from dataset import DukeDataset, OCTDataset, RESCDatasetSeg

# from network.unet_model import UNetWeakly, UNetWeaklyAtt
from network.final_net import IntegratedMixformer
from utils.utils_gradcam import CAMGeneratorAndSave


from utils.utils import (
    OrgLabels,
    get_gt_and_relevant_cams,
    save_models,
    save_tensorboard,
    normalized_batch_tensor,
)
from utils.metrics import (
    calculate_classification_metrics,
    calculate_roc,
    get_miou_for_multilevel_preds,
    scores,
    record_score,
)
from options import Configs
import numpy as np

import warnings

warnings.filterwarnings("ignore")

# print full columns
pd.set_option("display.max_columns", None)


class Train:
    def __init__(self, is_inference=False):
        self.orig_pretrained = not is_inference
        self.args = Configs().parse(is_inference)
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
        self.full_num_class = (
            self.num_class + 1 if "BackGround" not in OrgLabels else self.num_class
        )
        self.loss_cls_bce = (
            nn.BCEWithLogitsLoss()
        )  # no need sigmoid for this logits loss
        self.loss_layer_ce = nn.CrossEntropyLoss()  # no need softmax
        if self.args.n_layer_channels == 12:
            self.layer_const = "layer_prob"
        else:
            self.layer_const = "layer_img"

    def update_metrics(
        self,
        cls_pred_dicts,
        gt_dicts,
        pred_dicts,
        total_cls_loss,
        total_acc,
        loss_collect,
        labels,
        binary_labels,
    ):
        for k, v in cls_pred_dicts.items():
            if k not in gt_dicts:
                gt_dicts[k] = []
                pred_dicts[k] = []
                total_cls_loss[k] = 0
                total_acc[k] = Counter({"acc": 0, "f1m": 0, "f1mi": 0})
            if "layer" in k:
                list_of_cats = ["LayeNeg", "LayerPos"]
                prob_preds = self.softmax(v)
                gt = F.one_hot(binary_labels, num_classes=2)
            else:
                list_of_cats = OrgLabels
                prob_preds = self.sigmoid(v)
                gt = labels
            batch_accuracies_metrics = calculate_classification_metrics(
                prob_preds, gt, list_of_cats=list_of_cats
            )
            gt_dicts[k] += gt.cpu().tolist()
            pred_dicts[k] += prob_preds.cpu().tolist()

            total_cls_loss[k] += loss_collect[k].cpu().item()
            total_acc[k] += Counter(batch_accuracies_metrics)
        return gt_dicts, pred_dicts, total_cls_loss, total_acc

    def get_cls_loss(self, cls_pred_dicts, orig_gt, layer_gt):
        loss_collect = {k: 0 for k in cls_pred_dicts.keys()}
        for k, v in cls_pred_dicts.items():
            if "layer" in k:
                loss_collect[k] = self.loss_layer_ce(v, layer_gt)
            else:
                loss_collect[k] = self.loss_cls_bce(v, orig_gt)
        total_loss = sum(loss_collect.values())
        return total_loss, loss_collect

    def get_constraints_loss(self, constraint_dict, labels):
        visual_cam = constraint_dict["main-cam"]
        textual_cam_3 = constraint_dict["clip-l3-sim"]
        textual_cam_4 = constraint_dict["clip-l4-sim"]

        visual_cam_normalized = F.normalize(visual_cam, p=2, dim=[2, 3])
        textual_cam_4_normalized = F.normalize(textual_cam_4, p=2, dim=[2, 3])
        textual_cam_3_normalized = F.normalize(textual_cam_3, p=2, dim=[2, 3])

        rescaled_visual_cam = F.interpolate(
            visual_cam,
            size=(textual_cam_3.shape[2], textual_cam_3.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        rescaled_visual_cam_normalized = F.normalize(
            rescaled_visual_cam, p=2, dim=[2, 3]
        )
        rescaled_textual_cam_4 = F.interpolate(
            textual_cam_4,
            size=(textual_cam_3.shape[2], textual_cam_3.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        rescaled_textual_cam_4_normalized = F.normalize(
            rescaled_textual_cam_4, p=2, dim=[2, 3]
        )

        losses = []
        losses_rescale3 = []
        losses_rescaletxt = []
        for cls, n in enumerate(OrgLabels):
            if n == "BackGround":
                continue
            cls_targets = labels[:, cls].unsqueeze(1).unsqueeze(2)
            cls_loss = F.mse_loss(
                visual_cam_normalized[:, cls],
                textual_cam_4_normalized[:, cls],
                reduction="none",
            )
            cls_loss = cls_loss * cls_targets
            losses.append(cls_loss)
            cls_loss3 = F.mse_loss(
                rescaled_visual_cam_normalized[:, cls],
                textual_cam_3_normalized[:, cls],
                reduction="none",
            )
            cls_loss3 = cls_loss3 * cls_targets
            losses_rescale3.append(cls_loss3)
            cls_losstxt = F.mse_loss(
                rescaled_textual_cam_4_normalized[:, cls],
                textual_cam_3_normalized[:, cls],
                reduction="none",
            )
            cls_losstxt = cls_losstxt * cls_targets
            losses_rescaletxt.append(cls_losstxt)
        combined_loss = torch.stack(losses, dim=1)
        combined_rescale3_loss = torch.stack(losses_rescale3, dim=1)
        combined_rescaletxt_loss = torch.stack(losses_rescaletxt, dim=1)

        loss = combined_loss.sum() / (labels[:, 1:].sum() + 1e-6)
        loss_rescale3 = combined_rescale3_loss.sum() / (labels[:, 1:].sum() + 1e-6)
        loss_rescaletxt = combined_rescaletxt_loss.sum() / (labels[:, 1:].sum() + 1e-6)
        return loss + loss_rescaletxt + loss_rescale3

    def valid_once(self, epoch, testloader, multi_task_model):
        multi_task_model.eval()
        total_acc_val = {}
        total_cls_loss_val = {}
        gt_dicts = {}
        pred_dicts = {}
        total_disloss = 0

        cam_img_list, gradcam_img_list = (
            [],
            [],
        )
        for batch, data in tqdm(enumerate(testloader), total=len(testloader)):
            (
                image,
                labels,
                binary_labels,
                layer_img,
                layer_uncertain,
                caption_tensor,
                anomaly_diff,
            ) = (
                data["image"].to(self.device),
                data["labels"].to(self.device),
                data["binary_labels"].to(self.device),
                data[self.layer_const].to(self.device),
                data["layer_uncertain"].to(self.device),
                data["caption_tensor"].to(self.device),
                data["anomaly_diff"].to(self.device),
            )
            if self.args.add_abnormal:
                layer_img = normalized_batch_tensor(layer_img + anomaly_diff)
            comb_input = torch.cat([image, layer_img, layer_uncertain], dim=1)
            # _, ready_gradcam_pred = self.CamModel.get_cam_and_save(
            #     {
            #         "inputs": data,
            #         "input_tensor": comb_input,
            #     }
            # )
            # gradcam_img_list += ready_gradcam_pred
            with torch.no_grad():
                (cls_pred_dicts, cams_dict, constraint_dict) = multi_task_model(
                    comb_input, caption_input=caption_tensor
                )
                total_loss, loss_collect = self.get_cls_loss(
                    cls_pred_dicts, labels, binary_labels
                )
                if self.args.constraint_loss:
                    distillation_loss = self.get_constraints_loss(
                        constraint_dict, labels
                    ).item()
                    total_disloss += distillation_loss

                updated_cams_dicts = get_gt_and_relevant_cams(
                    cams_dict,
                    data,
                    self.args,
                )
                cam_img_list += updated_cams_dicts

                (
                    gt_dicts,
                    pred_dicts,
                    total_cls_loss_val,
                    total_acc_val,
                ) = self.update_metrics(
                    cls_pred_dicts,
                    gt_dicts,
                    pred_dicts,
                    total_cls_loss_val,
                    total_acc_val,
                    loss_collect,
                    labels,
                    binary_labels,
                )

        print(f"Num of CAM images: {len(cam_img_list)}")
        gt_for_cam = [x["gt"] for x in cam_img_list]
        gt_for_layer_cam = [(x["gt"] > 0) * 1 for x in cam_img_list]

        # score_gradcam = scores(
        #     gt_for_cam, gradcam_img_list, n_class=self.full_num_class
        # )
        # print("GradCAM:\n", score_gradcam)

        all_keys = get_miou_for_multilevel_preds(
            cam_img_list,
            gt_for_cam,
            gt_for_layer_cam,
            self.full_num_class,
            self.args.out_cam_pred_alpha,
        )
        print(pd.DataFrame(all_keys))
        final_cam_score = [x for x in all_keys if x["cam_n"] == "final_cam"][0]["score"]

        # Print accuracy
        if self.args.constraint_loss:
            total_disloss = round(total_disloss / (batch + 1), 4)
            print(f"Distillation Loss: {total_disloss}")
        valid_metrics_epoch = {}
        valid_roc_class_dict = {}
        for k in gt_dicts.keys():
            valid_metrics_epoch[k] = {
                "loss": round(total_cls_loss_val[k] / (batch + 1), 4),
                **{k: round(v / (batch + 1), 4) for k, v in total_acc_val[k].items()},
            }
            if "layer" in k:
                continue
            roc_avg, roc_class = calculate_roc(pred_dicts[k], gt_dicts[k], name=k)
            valid_roc_class_dict[k] = {**roc_class, "Avg": roc_avg}
        print(f"===== Validation Epoch {str(epoch + 1)} =====")
        print(f"===== Accuracy & F1 =====")
        valid_metrics_epoch_p = pd.DataFrame(valid_metrics_epoch).T
        if "LayerNeg" in valid_metrics_epoch_p:
            valid_metrics_epoch_p.drop(columns=["LayerNeg"], inplace=True)
        print(valid_metrics_epoch_p)
        print(f"===== ROC =====")
        print(pd.DataFrame(valid_roc_class_dict).T)

        return (
            valid_metrics_epoch,
            valid_roc_class_dict,
            final_cam_score,
        )

    def train_once(
        self, epoch, trainloader, multi_task_model, multi_optimizer, lr_scheduler
    ):
        multi_task_model.train()

        total_acc = {}
        total_cls_loss = {}
        gt_dicts = {}
        pred_dicts = {}
        total_disloss = 0
        for batch, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            (
                image,
                labels,
                binary_labels,
                layer_img,
                layer_uncertain,
                caption_tensor,
                anomaly_diff,
            ) = (
                data["image"].to(self.device),
                data["labels"].to(self.device),
                data["binary_labels"].to(self.device),
                data[self.layer_const].to(self.device),
                data["layer_uncertain"].to(self.device),
                data["caption_tensor"].to(self.device),
                data["anomaly_diff"].to(self.device),
            )
            if self.args.add_abnormal:
                layer_img = normalized_batch_tensor(layer_img + anomaly_diff)
            comb_input = torch.cat([image, layer_img, layer_uncertain], dim=1)
            multi_optimizer.zero_grad()
            (cls_pred_dicts, cams_dict, constraint_dict) = multi_task_model(
                comb_input, caption_input=caption_tensor
            )

            total_loss, loss_collect = self.get_cls_loss(
                cls_pred_dicts, labels, binary_labels
            )
            if self.args.constraint_loss:
                distillation_loss = self.get_constraints_loss(
                    constraint_dict, labels
                ).item()
                total_loss += distillation_loss
                total_disloss += distillation_loss

            total_loss.backward()
            multi_optimizer.step()
            lr_scheduler.step()

            ############## Metrics Calculation ################
            with torch.no_grad():
                gt_dicts, pred_dicts, total_cls_loss, total_acc = self.update_metrics(
                    cls_pred_dicts,
                    gt_dicts,
                    pred_dicts,
                    total_cls_loss,
                    total_acc,
                    loss_collect,
                    labels,
                    binary_labels,
                )
        if self.args.constraint_loss:
            total_disloss = round(total_disloss / (batch + 1), 4)
            print(f"Distillation Loss: {total_disloss}")

        train_metrics_epoch = {}
        roc_class_dict = {}
        for k in gt_dicts.keys():
            train_metrics_epoch[k] = {
                "loss": round(total_cls_loss[k] / (batch + 1), 4),
                **{k: round(v / (batch + 1), 4) for k, v in total_acc[k].items()},
            }
            if "layer" in k:
                continue
            roc_avg, roc_class = calculate_roc(pred_dicts[k], gt_dicts[k], name=k)
            roc_class_dict[k] = {**roc_class, "Avg": roc_avg}

        print(f"===== Epoch {str(epoch + 1)} =====")
        print(f"===== Accuracy & F1 =====")
        train_metrics_epoch_p = pd.DataFrame(train_metrics_epoch).T
        if "LayerNeg" in train_metrics_epoch_p:
            train_metrics_epoch_p.drop(columns=["LayerNeg"], inplace=True)
        print(train_metrics_epoch_p)
        print(f"===== ROC =====")
        print(pd.DataFrame(roc_class_dict).T)

        return train_metrics_epoch, roc_class_dict

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
    
    def get_clip_feature(self):
        if not self.args.clip_branch:
            return None
        if "our_dataset" in self.args.root_dirs:
            data_dir = "ours"
        elif "RESC" in self.args.root_dirs:
            data_dir = "resc"
        else:
            data_dir = "duke"
        if self.args.clip_version == "base":
            clip_f_path = f"text_features/{data_dir}/clip_label_base.pkl"
        elif self.args.clip_version == "large":
            clip_f_path = f"text_features/{data_dir}/clip_label_large.pkl"
        else:
            raise NotImplementedError

        with open(clip_f_path, "rb") as f:
            clip_f = pickle.load(f).cpu()
        if "BackGround" not in OrgLabels:
            # there is no background class
            clip_f = self.clip_f[1:]
        return clip_f

    def get_models(self):
        multi_task_model = IntegratedMixformer(
            layer_branch=self.args.layer_branch,
            clip_f=self.get_clip_feature(),
            img_size=self.args.is_size[0],
            cls_num_classes=self.num_class,
            stride=[4, 2, 2, 2],  # default 4 2 2 2
            pretrained=self.orig_pretrained,
            pool_type=self.args.pool_type,
            freeze_layers=self.args.freeze_layers,
            caption_version=self.args.caption_version,
            caption_branch=self.args.caption_branch,
            constraint_loss=self.args.constraint_loss,
        )
        print(multi_task_model)
        # self.CamModel = CAMGeneratorAndSave(self.args, multi_task_model)
        if self.args.load_model:
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
        non_frozen_params = [
            p for p in multi_task_model.parameters() if p.requires_grad
        ]
        if self.args.optimizer == "adam":
            multi_optimizer = optim.Adam(non_frozen_params, lr=self.args.lr)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                multi_optimizer,
                step_size=self.args.lr_schedule["step"],
                gamma=self.args.lr_schedule["gamma"],
            )
        elif self.args.optimizer == "adamw":
            multi_optimizer = optim.AdamW(
                params=non_frozen_params,
                lr=self.args.lr,
                weight_decay=0.001,
                betas=(0.9, 0.999),
            )
            lr_scheduler = get_polynomial_decay_schedule_with_warmup(
                multi_optimizer,
                num_warmup_steps=self.args.warmup_iter,
                num_training_steps=self.args.max_iters,
                power=1,
            )
        else:
            raise NotImplementedError

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

        iters_per_epoch = len(trainloader)
        self.args.max_iters = self.args.n_epochs * iters_per_epoch
        self.args.warmup_iter = self.args.warmup_ratio * self.args.max_iters

        multi_task_model, multi_optimizer, lr_scheduler = self.get_models()
        best_roc = 0
        best_mIoU = 0

        epoch = 0
        for epoch in range(0, self.args.n_epochs):
            (train_metrics_epoch, roc_class_dict) = self.train_once(
                epoch, trainloader, multi_task_model, multi_optimizer, lr_scheduler
            )
            train_tensorboard_dict = {
                "total_train_acc_matrix": train_metrics_epoch["main-cls"],
                "total_train_roc_matrix": roc_class_dict["main-cls"],
            }
            save_tensorboard(self.tb, train_tensorboard_dict, epoch, log_type="Train")
            # ######## Validation step per 5 epochs ##########
            if (epoch + 1) % 5 == 0:
                (
                    valid_metrics_epoch,
                    valid_roc_class_dict,
                    mIoU,
                ) = self.valid_once(epoch, testloader, multi_task_model)
                if valid_roc_class_dict["main-cls"]["Avg"] >= best_roc:
                    save_models(
                        self.args,
                        epoch,
                        multi_task_model,
                        multi_optimizer,
                        best_type="cls",
                    )
                    best_roc = valid_roc_class_dict["main-cls"]["Avg"]
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
                    "total_val_acc_matrix": valid_metrics_epoch["main-cls"],
                    "total_val_roc_matrix": valid_roc_class_dict["main-cls"],
                }
                save_tensorboard(self.tb, eval_tensorboard_dict, epoch, log_type="Val")

        save_models(self.args, epoch, multi_task_model, multi_optimizer)

        print("final running time:", time.time() - start)


if __name__ == "__main__":
    trainer = Train(is_inference=False)
    trainer.train()
