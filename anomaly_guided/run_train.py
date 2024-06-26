import os
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from network.utils import network_class
from utils.loss import dsc_loss
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE_NR = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NR

from gan_inference import load_gan_model
from dataset import DukeDataset, OCTDataset, RESCDataset
from refine_pseudo_label import refine_input_by_cam
from network.agm import AGM
from network.single_models import CNNs

from utils.utils import (
    CAMGeneratorAndSave,
    OrgLabels,
    diff_map_for_att,
    get_num_classes,
    save_models,
    save_tensorboard_train,
    save_tensorboard_val,
)
from utils.metrics import (
    calculate_classification_metrics,
    calculate_roc,
    scores,
    record_score,
)
from options import Configs


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
        self.backbone = network_class(self.args)
        self.num_class = len(OrgLabels)
        self.gan_pretrained = False
        with torch.no_grad():
            if self.args.input_gan:
                path = "{}/netG.pth".format(self.args.model_gan)
                self.gan_pretrained = load_gan_model(path, DEVICE_NR)
                print(f" Loaded Pretained GAN weights from {path}.")
        self.transform_norml = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.w_ce = self.args.w_ce
        self.w_dice = self.args.w_dice

        self.loss_cls_func = (
            nn.BCEWithLogitsLoss()
        )  # no need sigmoid for this logits loss

        self.include_refinement = False
        self.checking_grad = torch.Tensor([0])

    def get_cls_loss(self, y_pred, y_true):
        loss_ce = self.loss_cls_func(y_pred, y_true)
        loss_dice = dsc_loss(y_pred, y_true)
        return self.w_ce * loss_ce + self.w_dice * loss_dice

    def valid_once(self, epoch, testloader, multi_task_model):
        multi_task_model.eval()
        total_acc_val = Counter({"acc": 0, "f1m": 0, "f1mi": 0})
        total_cls_loss_val = 0
        gt_list = []
        pred_list = []

        CAMGenerationModule = CAMGeneratorAndSave(
            opts=self.args, multi_task_model=multi_task_model
        )
        CAMGenerationModule.set_epoch(epoch)
        gt_img_list, cam_img_list = [], []
        tensor_for_att = None
        for batch, data in tqdm(enumerate(testloader), total=len(testloader)):
            image, labels, mask = (
                data["image"].to(self.device),
                data["labels"].to(self.device),
                data["mask"].to(self.device),
            )
            updated_image = image.clone()
            with torch.no_grad():
                if self.args.input_gan:
                    gan_inputs = self.transform_norml(updated_image)
                    healthy_img = self.gan_pretrained.inference(gan_inputs)
                    if self.args.att_module:
                        tensor_for_att = diff_map_for_att(
                            updated_image, healthy_img, mask
                        )
                        updated_image = torch.cat(
                            (image, healthy_img, tensor_for_att), dim=1
                        )
                    else:
                        updated_image = torch.cat((image, healthy_img), dim=1)

            if self.include_refinement:
                updated_image = refine_input_by_cam(
                    self.device, multi_task_model, updated_image, mask
                )
            with torch.no_grad():
                cls_outputs = multi_task_model(updated_image)
                sig_prediction = self.sigmoid(cls_outputs)

            params = {
                "inputs": data,
                "batch_preds": sig_prediction,
                "refined": updated_image,
            }
            gt_res, pred_res = CAMGenerationModule.get_cam_and_save(params)
            gt_img_list += gt_res
            cam_img_list += pred_res

            with torch.no_grad():
                clf_loss = self.get_cls_loss(cls_outputs, labels)
                total_cls_loss_val += clf_loss.cpu().item()
                batch_accuracies_metrics = calculate_classification_metrics(
                    sig_prediction, labels
                )
                total_acc_val += Counter(batch_accuracies_metrics)
                gt_list += labels.cpu().tolist()
                pred_list += sig_prediction.tolist()
        print(f"Num of CAM images: {len(pred_list)}")
        score = scores(gt_img_list, cam_img_list, n_class=get_num_classes() + 1)
        print(score)
        record_score(score, self.iou_txt_path)

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
        return valid_loss_epoch, valid_acc_epoch, roc_avg, roc_class, score["Mean IoU"]

    def train_once(
        self, epoch, trainloader, multi_task_model, multi_optimizer, lr_scheduler
    ):
        multi_task_model.train()
        tensor_for_att = None

        total_acc = Counter({"acc": 0, "f1m": 0, "f1mi": 0})
        total_cls_loss = 0
        gt_list, pred_list = [], []
        for batch, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            image, labels, mask = (
                data["image"].to(self.device),
                data["labels"].to(self.device),
                data["mask"].to(self.device),
            )
            updated_image = image.clone()

            ############## GAN healthy generation ################
            if self.args.input_gan:
                with torch.no_grad():
                    gan_inputs = self.transform_norml(updated_image)
                    healthy_img = self.gan_pretrained.inference(gan_inputs)

                ############## Activate Attention Branch ################
                if self.args.att_module:
                    tensor_for_att = diff_map_for_att(updated_image, healthy_img, mask)
                    updated_image = torch.cat(
                        (image, healthy_img, tensor_for_att), dim=1
                    )
                else:
                    updated_image = torch.cat((image, healthy_img), dim=1)

            ############## CAM refinement step ################
            if self.include_refinement:
                updated_image = refine_input_by_cam(
                    self.device, multi_task_model, updated_image, mask
                )

            multi_optimizer.zero_grad()
            cls_outputs = multi_task_model(updated_image)
            sig_prediction = self.sigmoid(cls_outputs)

            combine_loss = self.get_cls_loss(cls_outputs, labels)
            combine_loss.backward()
            multi_optimizer.step()

            ############## Metrics Calculation ################
            with torch.no_grad():
                total_cls_loss += combine_loss.cpu().item()
                batch_accuracies_metrics = calculate_classification_metrics(
                    sig_prediction, labels
                )
                total_acc += Counter(batch_accuracies_metrics)
                gt_list += labels.cpu().tolist()
                pred_list += sig_prediction.cpu().tolist()
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
            dataset_train = RESCDataset(self.args, data_type="train")
            dataset_test = RESCDataset(self.args, data_type="test")
        else:
            dataset_train = DukeDataset(self.args, data_type="train")
            dataset_test = DukeDataset(self.args, data_type="test")

        return dataset_train, dataset_test

    def get_models(self):
        if self.args.retinal_model == "AGM":
            retinal_model_func = AGM
        elif self.args.retinal_model == "CNNs":
            retinal_model_func = CNNs
        else:
            raise NotImplementedError()
        multi_task_model = retinal_model_func(
            self.backbone,
            num_class=self.num_class,
            num_input_channel=self.num_input_channel,
            backbone_name=self.args.backbone,
        )
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
        multi_optimizer = optim.SGD(
            multi_task_model.parameters(), lr=self.args.lr, momentum=0.9
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            multi_optimizer,
            step_size=self.args.lr_schedule["step"],
            gamma=self.args.lr_schedule["gamma"],
        )

        return multi_task_model, multi_optimizer, lr_scheduler

    def train(self):
        start = time.time()
        dataset_train, dataset_test = self.get_dataset()
        self.num_input_channel = (
            dataset_train[0]["image"].shape[0] * 2
            if self.args.input_gan
            else dataset_train[0]["image"].shape[0]
        )
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
        # TODO: the regular training epoch must consider the continue train as well...
        # Regular training process
        epoch = 0
        for epoch in range(0, self.args.reg_epochs):
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
            if epoch == self.args.reg_epochs - 1:
                ## Save model for last regular train epoch
                save_models(self.args, epoch, multi_task_model, multi_optimizer)

        ref_begin_ep = self.pretrained_epoch if self.args.continue_train else epoch
        # Refinement CAM training process
        for r_epoch in range(0, self.args.iter_epochs):
            self.include_refinement = True
            curr_epoch = ref_begin_ep + r_epoch
            (
                train_loss,
                train_acc_matrix,
                train_roc_avg,
                train_roc_class,
            ) = self.train_once(
                curr_epoch, trainloader, multi_task_model, multi_optimizer, lr_scheduler
            )
            train_tensorboard_dict = {
                "total_train_loss": train_loss,
                "total_train_acc_matrix": train_acc_matrix,
                "total_train_roc": train_roc_avg,
                "total_train_roc_matrix": train_roc_class,
            }
            save_tensorboard_train(self.tb, train_tensorboard_dict, curr_epoch)
            ######## Validate for each epoch in refinement steps ##########
            (
                valid_loss,
                valid_acc_matrxi,
                valid_roc_avg,
                valid_roc_class,
                _,
            ) = self.valid_once(curr_epoch, testloader, multi_task_model)
            eval_tensorboard_dict = {
                "total_val_loss": valid_loss,
                "total_val_acc_matrix": valid_acc_matrxi,
                "total_val_roc": valid_roc_avg,
                "total_val_roc_matrix": valid_roc_class,
            }
            save_tensorboard_val(self.tb, eval_tensorboard_dict, curr_epoch)
            save_models(
                self.args, curr_epoch, multi_task_model, multi_optimizer, is_iter="iter"
            )

        print("final running time:", time.time() - start)


if __name__ == "__main__":
    trainer = Train()
    trainer.train()
