import os
import torch
import numpy as np
from options_seg import Configs

import cv2

import torch.nn.functional as F
from PIL import Image
import torch.nn as nn

type_color = {
    0: [0, 0, 0],  # black
    1: [255, 0, 0],  # srf red
    2: [0, 255, 0],  # irf green
    3: [0, 0, 255],  # ez blue
    4: [255, 255, 0],  # hrd yellow
    5: [255, 165, 0],  # rpe orange
    6: [255, 0, 255],  # back ground pink
}
OrgLabels = (
    Configs().get_labels()
)  # ['BackGround', 'SRF', 'IRF', 'EZ', 'HRD',  'RPE', ]


CAT_LIST = ["background"] + [x for x in OrgLabels if x != "BackGround"] + ["meanIOU"]


def get_mask_path_by_image_path(image_path):
    if "BOE" in image_path:
        mask_path = image_path.replace("/images/", "/mask/")
    elif "RESC" in image_path:
        mask_path = image_path.replace("/RESC/", "/RESC/mask/").replace(
            "original_images/", ""
        )
    elif "NORMAL" in image_path:
        mask_path = image_path.replace("train/0.normal", "normal_mask")
    else:
        mask_path = image_path.replace("original", "mask")
    return mask_path


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_swim(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform(tensor):
    if tensor.shape[1] != 197 and tensor.shape[1] != 49:
        return tensor
    if tensor.shape[1] == 197:
        return reshape_transform_vit(tensor)
    return reshape_transform_swim(tensor)


def get_num_classes():
    if "BackGround" in OrgLabels:
        return len(OrgLabels) - 1
    return len(OrgLabels)


def normalized_batch_tensor(t):
    orig_size = t.shape
    t = t.view(orig_size[0], -1)
    t -= t.min(1, keepdim=True)[0]
    t /= t.max(1, keepdim=True)[0]
    t = t.view(orig_size)
    return t


def diff_map_for_att(orig_tensor, gan_tensor, mask_tensor):
    # batch, channel, h, w
    normalized_orig = orig_tensor.clone()
    normalized_gan = gan_tensor.clone()

    normalized_orig = normalized_batch_tensor(normalized_orig)
    normalized_gan = normalized_batch_tensor(normalized_gan)

    abs_diff = torch.abs(normalized_orig - normalized_gan)
    mask_out_diff = abs_diff * mask_tensor

    return mask_out_diff


# input is array H * W
def convert_resc_pix2image_label(label_pixel_arr):
    # back: 0, ped: 128, srf: 191, retinal: 255
    image_label = {"SRF": 0, "PED": 0, "lesion": 0, "health": 0, "BackGround": 1}
    labels = np.unique(label_pixel_arr)
    if len(labels) == 1:
        image_label["health"] += 1
    if 128 in labels:
        image_label["PED"] += 1
    if 191 in labels:
        image_label["SRF"] += 1
    if 255 in labels:
        image_label["lesion"] += 1
    return image_label


def convert_resc_labels(img):
    # 0 background,
    # 1 lesion(need turn to background),  1 -> 0
    # 0.74 SRF(need turn to 1),           0.74 -> 1
    # 0.51 PED(need turn to 2)            0.51 -> 2
    # back: 0, ped: 128, srf: 191, retinal: 255
    img[img == 255] = 0
    img[img == 191] = 1
    img[img == 128] = 2
    return img


def convert_duke_labels(img):
    img[img < 255] = 0
    img[img == 255] = 1
    return img


def convert_our_dataset_labels(img):
    # "categories":[{"id":51,"name":"IRF"},{"id":102,"name":"SRF"},{"id":153,"name":"HRD"},{"id":204,"name":"EZ disruption"},{"id":255,"name":"RPE "}]}
    # ['SRF', 'IRF', 'EZ disrupted', 'HRD', 'BackGround']
    img[img == 51] = 2
    img[img == 102] = 1
    img[img == 153] = 4
    img[img == 204] = 3
    img[img == 255] = 0  # we dont include RPE for now

    return img


def convert_data_labels(img, root_dirs):
    if "RESC" in root_dirs:
        return convert_resc_labels(img)
    if "BOE" in root_dirs:
        return convert_duke_labels(img)
    return convert_our_dataset_labels(img)


def post_process_cam(resized_cam, orig_mask=None):
    masked_cam = resized_cam
    if orig_mask is not None:
        mask_clone = orig_mask.copy()  # NEUROSENSORY RETINA only (ILM to RPE)
        mask_clone[mask_clone == 0] = 0.5
        masked_cam = resized_cam * mask_clone

    # cam_max = masked_cam.max()
    # cam_min = masked_cam.min()
    cam_max = np.max(masked_cam, (1, 2), keepdims=True)
    cam_min = np.min(masked_cam, (1, 2), keepdims=True)
    masked_cam[masked_cam < cam_min + 1e-5] = 0

    norm_masked_cam = (masked_cam - cam_min - 1e-5) / (cam_max - cam_min + 1e-5)

    return norm_masked_cam


def get_seg_and_save(path, seg_out, labels, epoch, opts, save_results=False):
    non_background_names = [x for x in OrgLabels if "BackGround" != x]
    lesion_classes = [OrgLabels.index(name) for name in non_background_names]
    # batch_preds [BC] B: batch, C: Class

    ready_pred_4d = []
    gt = []
    for batch_nb, seg_pred in enumerate(seg_out):
        ground_true_classes = [i for i, v in enumerate(labels[batch_nb]) if v > 0.5]
        # only calculate and save for ground truth lesion images
        if not len(set(lesion_classes) & set(ground_true_classes)):
            continue
        image_path = path[batch_nb]
        img_name = image_path.split("/")[-1]
        # 0-255 pixel value, numpy
        orig_img = np.asarray(Image.open(image_path))
        if len(orig_img.shape) == 3:
            orig_img = orig_img[..., 0]

        if "our_dataset" in opts.root_dirs:
            if opts.expert_annot == "both":
                annot_path = os.path.join(
                    opts.root_dirs, opts.annot_dir, img_name.split(".")[0] + ".png"
                )
            else:
                expert_annot_img_name = (
                    img_name.split(".")[0] + f"_{opts.expert_annot}.png"
                )
                annot_path = os.path.join(
                    opts.root_dirs, opts.annot_dir, expert_annot_img_name
                )
        else:
            annot_path = os.path.join(
                opts.root_dirs, opts.annot_dir, img_name
            )  # resc valid/label_images
        orig_annot = np.asarray(Image.open(annot_path))

        truth_label = [OrgLabels[cls] for cls in ground_true_classes]
        truth_label = "_".join(truth_label)
        seg_prob_map = normalized_batch_tensor(seg_pred.clone())
        resize_seg = F.interpolate(
            seg_prob_map.unsqueeze(0),  # [1, cls, w, h]
            size=(orig_img.shape[0], orig_img.shape[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            0
        )  # [cls, w, h]
        # normalize to 0-1 again
        # resize_seg = post_process_cam(resize_seg.squeeze().detach().cpu().numpy())
        # resize_seg = np.argmax(resize_seg, axis=0)
        # import pdb; pdb.set_trace()

        # convert non-lesion classes channel to 0
        non_lesion_classes = [i for i, v in enumerate(labels[batch_nb]) if v < 0.5]
        resize_seg[non_lesion_classes] = 0

        binarized_seg = resize_seg.argmax(0)
        save_seg = (
            torch.clamp(binarized_seg / get_num_classes(), 0, 1)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        save_seg = np.uint8(save_seg * 255)
        # import pdb; pdb.set_trace()

        save_img = [orig_img, orig_annot, save_seg]
        save_image_h = cv2.hconcat(save_img)

        gt_labels = convert_data_labels(orig_annot, opts.root_dirs)

        gt.append(gt_labels)
        ready_pred_4d.append(binarized_seg.detach().cpu().numpy())

        if not save_results:
            continue
        save_path = os.path.join(
            opts.save_folder, "images", "{}".format(img_name.split(".")[0])
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(
            save_path + "/epoch{}_{}.jpg".format(epoch, truth_label), save_image_h
        )
    # import pdb; pdb.set_trace()
    return gt, ready_pred_4d


def restore_and_save(path, seg_out, labels, epoch, opts, save_results=False):
    non_background_names = [x for x in OrgLabels if "BackGround" != x]
    lesion_classes = [OrgLabels.index(name) for name in non_background_names]
    # batch_preds [BC] B: batch, C: Class

    ready_pred_4d = []
    gt = []
    for batch_nb, seg_pred in enumerate(seg_out):
        ground_true_classes = labels[batch_nb].unique().tolist()
        # only calculate and save for ground truth lesion images
        if not len(set(lesion_classes) & set(ground_true_classes)):
            continue
        image_path = path[batch_nb]
        img_name = image_path.split("/")[-1]
        # 0-255 pixel value, numpy
        orig_img = np.asarray(Image.open(image_path))
        if len(orig_img.shape) == 3:
            orig_img = orig_img[..., 0]

        if "our_dataset" in opts.root_dirs:
            if opts.expert_annot == "both":
                annot_path = os.path.join(
                    opts.root_dirs, opts.annot_dir, img_name.split(".")[0] + ".png"
                )
            else:
                expert_annot_img_name = (
                    img_name.split(".")[0] + f"_{opts.expert_annot}.png"
                )
                annot_path = os.path.join(
                    opts.root_dirs, opts.annot_dir, expert_annot_img_name
                )
        else:
            annot_path = os.path.join(
                opts.root_dirs, opts.annot_dir, img_name
            )  # resc valid/label_images
        orig_annot = np.asarray(Image.open(annot_path))

        truth_label = [OrgLabels[cls] for cls in ground_true_classes]
        truth_label = "_".join(truth_label)
        # seg_prob_map = normalized_batch_tensor(seg_pred.clone())
        # import pdb; pdb.set_trace()
        seg_prob_map = F.softmax(seg_pred.clone(), dim=0)  # log_softmax?
        resize_seg = F.interpolate(
            seg_prob_map.unsqueeze(0),  # [1, cls, w, h]
            size=(orig_img.shape[0], orig_img.shape[1]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            0
        )  # [cls, w, h]

        binarized_seg = resize_seg.argmax(0)
        save_seg = (
            torch.clamp(binarized_seg / get_num_classes(), 0, 1)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        save_seg = np.uint8(save_seg * 255)
        # import pdb; pdb.set_trace()

        save_img = [orig_img, orig_annot, save_seg]
        save_image_h = cv2.hconcat(save_img)

        gt_labels = convert_data_labels(orig_annot, opts.root_dirs)

        gt.append(gt_labels)
        ready_pred_4d.append(binarized_seg.detach().cpu().numpy())

        if not save_results:
            continue
        save_path = os.path.join(
            opts.save_folder, "images", "{}".format(img_name.split(".")[0])
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(
            save_path + "/epoch{}_{}.jpg".format(epoch, truth_label), save_image_h
        )
    # import pdb; pdb.set_trace()
    return gt, ready_pred_4d


def save_models(
    args, epoch, multi_task_model, multi_optimizer, best_type=None, is_iter=False
):
    save_path = f"./{args.save_folder}/weights"
    if best_type is None:
        save_name = epoch + 1
    elif best_type == "pseudo":
        save_name = "best_iou"
    elif best_type == "cls":
        save_name = "best"
    else:
        raise ValueError("The type for save model is not available")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if is_iter:
        save_name = f"iter_{save_name}"
    torch.save(
        {
            "epoch": epoch,
            "args": args,
            "state_dict": multi_task_model.state_dict(),
            "optimizer": multi_optimizer.state_dict(),
        },
        save_path + "/{0}.pwf".format(save_name),
    )


def save_tensorboard_val(tb, loss_dict, mark_epoch):
    tb.add_scalar("Loss/Valid", loss_dict["total_val_loss"], mark_epoch + 1)
    tb.add_scalar("ROC/Valid", loss_dict["total_val_roc"], mark_epoch + 1)
    for acc_type in ["acc", "f1m"]:
        tb.add_scalar(
            "Val Accuracy/{}".format(acc_type),
            loss_dict["total_val_acc_matrix"][acc_type],
            mark_epoch + 1,
        )
    for label_type in OrgLabels:
        tb.add_scalar(
            "Val Class Acc/{}".format(label_type),
            loss_dict["total_val_acc_matrix"][label_type],
            mark_epoch + 1,
        )
        tb.add_scalar(
            "Val Class ROC/{}".format(label_type),
            loss_dict["total_val_roc_matrix"][label_type],
            mark_epoch + 1,
        )
    tb.close()


def save_tensorboard_train(tb, loss_dict, mark_epoch):
    tb.add_scalar("Loss/Train", loss_dict["total_train_loss"], mark_epoch + 1)
    tb.add_scalar("ROC/Train", loss_dict["total_train_roc"], mark_epoch + 1)
    for acc_type in ["acc", "f1m"]:
        tb.add_scalar(
            "Train Accuracy/{}".format(acc_type),
            loss_dict["total_train_acc_matrix"][acc_type],
            mark_epoch + 1,
        )
    for label_type in OrgLabels:
        tb.add_scalar(
            "Train Class Acc/{}".format(label_type),
            loss_dict["total_train_acc_matrix"][label_type],
            mark_epoch + 1,
        )
        tb.add_scalar(
            "Train Class ROC/{}".format(label_type),
            loss_dict["total_train_roc_matrix"][label_type],
            mark_epoch + 1,
        )
    tb.close()


class LossFunctions:
    def __init__(self) -> None:
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def get_loss(
        self,
        y_true,
        y_pred,
        loss_type="ce",
        layer_pred=None,
        lesion_in_layer=None,
        scale_factor=None,
        threshold=0.7,
    ):
        if loss_type == "ce":
            return self.get_ce_loss(y_pred, y_true)
        elif loss_type == "layer":
            return self.get_layer_loss(
                y_true, y_pred, layer_pred, lesion_in_layer, scale_factor, threshold
            )
        elif loss_type == "layer_ped":
            return self.get_layer_ped_loss(
                y_true, y_pred, layer_pred, lesion_in_layer, scale_factor, threshold
            )
        raise ValueError("The loss type is not available")

    def get_ce_loss(self, y_pred, y_true):
        """
        y_true: ground truth lesion segmentation, shape: [batch, 512, 512]
        y_pred: predicted lesion segmentation logits, shape: [batch, num_classes, 512, 512]
        """
        return self.cross_entropy_loss(y_pred, y_true.long())

    def get_layer_loss(
        self, y_true, y_pred, layer_pred, lesion_in_layer, scale_factor, threshold
    ):
        """
        y_true: ground truth lesion segmentation, shape: [batch, 512, 512]
        y_pred: predicted lesion segmentation logits, shape: [batch, num_classes, 512, 512]
        layer_pred: predicted layer segmentation logits, shape: [batch, num_layers, 512, 512]
        lesion_in_layer: a dictionary indicating which lesions can occur in which layers
        scale_factor: a factor to scale the penalty for invalid predictions
        """
        # Standard cross entropy loss for lesion segmentation
        lesion_loss = F.cross_entropy(
            y_pred, y_true.long(), ignore_index=-1, reduction="none"
        )

        # Transpose y_pred and layer_pred to match y_true shape
        y_pred_softmax = F.softmax(y_pred, dim=1).permute(0, 2, 3, 1)
        layer_pred_softmax = F.softmax(layer_pred, dim=1).permute(0, 2, 3, 1)

        for lesion_class, valid_layers in lesion_in_layer.items():
            # background class should not be penalized - non-lesion can happen in any layer
            if lesion_class == 0:
                continue
            for layer in range(layer_pred_softmax.shape[-1]):
                # If the lesion can occur in this layer, continue
                if layer in valid_layers:
                    continue
                # import pdb; pdb.set_trace()

                # Generate a mask for the pixels where the lesion is predicted but shouldn't occur in this layer
                invalid_mask = (
                    torch.argmax(y_pred_softmax, dim=-1) == lesion_class
                ) & (layer_pred_softmax[..., layer] > threshold)

                # Add to the loss the product of the predicted lesion and layer probabilities for these pixels, scaled by the scale factor
                lesion_loss[invalid_mask] += (
                    scale_factor
                    * y_pred_softmax[invalid_mask][:, lesion_class]
                    * layer_pred_softmax[invalid_mask][:, layer]
                )

        return lesion_loss.mean()

    def get_layer_ped_loss(
        self, y_true, y_pred, layer_pred, lesion_in_layer, scale_factor, threshold
    ):
        """
        y_true: ground truth lesion segmentation, shape: [batch, 512, 512]
        y_pred: predicted lesion segmentation logits, shape: [batch, num_classes, 512, 512]
        layer_pred: predicted layer segmentation logits, shape: [batch, num_layers, 512, 512]
        lesion_in_layer: a dictionary indicating which lesions can occur in which layers
        scale_factor: a factor to scale the penalty for invalid predictions
        """
        # Standard cross entropy loss for lesion segmentation
        lesion_loss = F.cross_entropy(
            y_pred, y_true.long(), ignore_index=-1, reduction="none"
        )

        # Transpose y_pred and layer_pred to match y_true shape
        y_pred_softmax = F.softmax(y_pred, dim=1).permute(0, 2, 3, 1)
        layer_pred = layer_pred.permute(0, 2, 3, 1)

        for lesion_class, valid_layers in lesion_in_layer.items():
            # background class should not be penalized - non-lesion can happen in any layer
            if lesion_class in [0, 1]:
                continue
            for layer in range(layer_pred.shape[-1]):
                # If the lesion can occur in this layer, continue
                if layer in valid_layers:
                    continue

                # Generate a mask for the pixels where the lesion is predicted but shouldn't occur in this layer
                invalid_mask = (
                    torch.argmax(y_pred_softmax, dim=-1) == lesion_class
                ) &  (torch.argmax(layer_pred, dim=-1) == layer) # (layer_pred[..., layer] > threshold) #

                # Add to the loss the product of the predicted lesion and layer probabilities for these pixels, scaled by the scale factor
                lesion_loss[invalid_mask] += (
                    scale_factor
                    * y_pred_softmax[invalid_mask][:, lesion_class]
                    * layer_pred[invalid_mask][:, layer]
                )

        return lesion_loss.mean()