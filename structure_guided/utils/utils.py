import os
import torch
import numpy as np
from options import Configs
import cv2

import time
import torchvision.models as models
import torchvision.utils as vutils
import torch.nn.functional as F
from PIL import Image

OrgLabels = Configs().get_labels()  # ['BackGround', 'SRF', 'IRF', 'EZ', 'HRD',  'RPE']


def get_annot_by_dataset(opts, img_name):
    if "our_dataset" in opts.root_dirs:
        if opts.expert_annot == "both":
            annot_path = os.path.join(
                opts.root_dirs, opts.annot_dir, img_name.split(".")[0] + ".png"
            )
        else:
            expert_annot_img_name = img_name.split(".")[0] + f"_{opts.expert_annot}.png"
            annot_path = os.path.join(
                opts.root_dirs, opts.annot_dir, expert_annot_img_name
            )
    else:
        annot_path = os.path.join(
            opts.root_dirs, opts.annot_dir, img_name
        )  # resc valid/label_images
    orig_annot = np.asarray(Image.open(annot_path))
    return orig_annot


def get_num_lesions():
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


def diff_map_for_att(orig_tensor, gan_tensor, mask_tensor=None):
    # batch, channel, h, w
    normalized_orig = orig_tensor.clone()
    normalized_gan = gan_tensor.clone()

    normalized_orig = normalized_batch_tensor(normalized_orig)
    normalized_gan = normalized_batch_tensor(normalized_gan)

    abs_diff = torch.abs(normalized_orig - normalized_gan)
    if mask_tensor is None:
        return abs_diff

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
    if len(OrgLabels) == 2 and "BackGround" in OrgLabels:
        img[img == 2] = 1
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
    norm_img = img.copy()
    if "RESC" in root_dirs:
        return convert_resc_labels(norm_img)
    if "BOE" in root_dirs:
        return convert_duke_labels(norm_img)
    return convert_our_dataset_labels(norm_img)


def get_resc_train_annot(img_name):
    annot_path = os.path.join("datasets/RESC/train/label_images", img_name)
    orig_annot = np.asarray(Image.open(annot_path))
    return convert_resc_labels(orig_annot)


def normalize_cam(cams_tensor):
    norm_cams = cams_tensor.detach().cpu().numpy()  # [batch, class, h, w]
    cam_max = np.max(norm_cams, (2, 3), keepdims=True)
    cam_min = np.min(norm_cams, (2, 3), keepdims=True)

    norm_cams = (norm_cams - cam_min) / (cam_max - cam_min + 1e-7)
    return norm_cams


def normalize_cam_tensor(cams_tensor):
    norm_cams = cams_tensor.detach().cpu()  # [batch, class, h, w]
    cam_max = torch.amax(norm_cams, (2, 3), keepdim=True)
    cam_min = torch.amin(norm_cams, (2, 3), keepdim=True)

    norm_cams = (norm_cams - cam_min) / (cam_max - cam_min + 1e-7)
    return norm_cams


def get_target_cams(cams_tensor, labels):
    norm_cams = normalize_cam(cams_tensor)
    label_masks = labels[:, 1:].unsqueeze(2).unsqueeze(3).detach().cpu().numpy()
    masked_cams = norm_cams * label_masks  # only keep the cam of the gt class
    return masked_cams


def get_target_cams_tensor(k, cams_tensor, labels, binary_labels):
    if "BackGround" in OrgLabels:
        updated_labels = labels[:, 1:]
    else:
        updated_labels = labels
    if "layer" in k:
        updated_labels = F.one_hot(binary_labels, num_classes=2)[:, 1:]
    norm_cams = normalize_cam_tensor(cams_tensor)
    label_masks = updated_labels.unsqueeze(2).unsqueeze(3).detach().cpu()
    masked_cams = norm_cams * label_masks  # only keep the cam of the gt class
    return masked_cams


def post_process_cam(cams_tensor, labels, bg_score=0.7):
    masked_cams = get_target_cams(cams_tensor, labels)
    bg_cam = (
        np.ones((cams_tensor.shape[0], 1, cams_tensor.shape[2], cams_tensor.shape[3]))
        * bg_score
    )
    cams_with_bg_score = torch.tensor(np.concatenate((bg_cam, masked_cams), axis=1))
    pred_cam_labels = torch.argmax(cams_with_bg_score, axis=1)
    return cams_with_bg_score, pred_cam_labels


def get_binarized_cam_pred(thres, t_name, cam_dict):
    cam = cam_dict[t_name]
    # cam: [cls, w, h] numpy
    bg_score = np.ones((1, cam[0].shape[0], cam[0].shape[1])) * thres / 100
    b_pred = np.argmax(np.concatenate((bg_score, cam), axis=0), axis=0).astype(np.uint8)
    return b_pred


def get_gt_and_relevant_cams(cams_dict, input_data, opts):
    path = input_data["path"]
    binary_labels = input_data["binary_labels"]
    multi_labels = input_data["labels"]
    # batch_preds [BC] B: batch, C: Class
    cams_dict = {
        k: get_target_cams_tensor(k, v, multi_labels, binary_labels)
        for k, v in cams_dict.items()
    }
    updated_cams_dicts = []
    for batch_nb in range(len(binary_labels)):
        # only calculate and save for ground truth lesion images
        if binary_labels[batch_nb] == 0:
            continue
        img_name = path[batch_nb].split("/")[-1]
        # 0-255 pixel value, numpy
        orig_annot = get_annot_by_dataset(opts, img_name)
        # import pdb; pdb.set_trace()
        # orig_annot = np.asarray(Image.open(path[batch_nb]))
        # import pdb; pdb.set_trace()
        assert len(orig_annot.shape) == 2
        original_size = (orig_annot.shape[0], orig_annot.shape[1])
        gt_labels = convert_data_labels(orig_annot, opts.root_dirs)
        updated_single_cam = {"img_name": img_name, "gt": gt_labels}
        for k, v in cams_dict.items():
            updated_single_cam[k] = (
                F.interpolate(
                    v[batch_nb].unsqueeze(0),  # [1, cls, w, h]
                    size=original_size,
                    mode="bilinear",
                    align_corners=True,
                )
                .cpu()
                .numpy()[0]
            )
        if "clip-l4-relu-sim" not in updated_single_cam:
            updated_single_cam["final_cam"] = updated_single_cam["main-relu-cam"]
        else:
            updated_single_cam["final_cam"] = (
                0.3 * updated_single_cam["clip-l3-relu-sim"] # 0.3
                + 0.3 * updated_single_cam["clip-l4-relu-sim"] # 0.3
                + 0.4 * updated_single_cam["main-relu-cam"] # 0.4
            )
        # cam_max = np.max(updated_single_cam["final_cam"], (1, 2), keepdims=True)
        # cam_min = np.min(updated_single_cam["final_cam"], (1, 2), keepdims=True)
        # updated_single_cam["final_cam"] = (
        #     updated_single_cam["final_cam"] - cam_min
        # ) / (cam_max - cam_min + 1e-5)

        # save npy
        # final_cams = updated_single_cam["final_cam"]
        # cam_save_name = img_name.split(".")[0]
        # np.save(f"best_cams/nyu/{cam_save_name}.npy", final_cams)
        # save pseudo label
        # final_cams = updated_single_cam["final_cam"]
        # cam_save_name = img_name.split(".")[0]
        # cam_save_name = path[batch_nb].replace("../","").replace("/","+").rsplit(".",1)[0]
        # np.save(f"pseudo_labels/duke/{cam_save_name}.npy", final_cams)
        # import pdb; pdb.set_trace()

        updated_cams_dicts.append(updated_single_cam)

    return updated_cams_dicts


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
        save_name = f"{is_iter}_{save_name}"
    torch.save(
        {
            "epoch": epoch,
            "args": args,
            "state_dict": multi_task_model.state_dict(),
            "optimizer": multi_optimizer.state_dict(),
        },
        save_path + "/{0}.pwf".format(save_name),
    )


def save_tensorboard(tb, loss_dict, mark_epoch, log_type):
    acc_name = f"total_{log_type.lower()}_acc_matrix"
    roc_name = f"total_{log_type.lower()}_roc_matrix"
    tb.add_scalar("Loss/Train", loss_dict[acc_name]["loss"], mark_epoch + 1)
    tb.add_scalar("ROC/Train", loss_dict[roc_name]["Avg"], mark_epoch + 1)
    for acc_type in ["acc", "f1m"]:
        tb.add_scalar(
            f"{log_type} Accuracy/{acc_type}",
            loss_dict[acc_name][acc_type],
            mark_epoch + 1,
        )
    for label_type in OrgLabels or "Layer" in label_type:
        tb.add_scalar(
            f"{log_type} Class Acc/{label_type}",
            loss_dict[acc_name][label_type],
            mark_epoch + 1,
        )
        tb.add_scalar(
            f"{log_type} Class ROC/{label_type}",
            loss_dict[roc_name][label_type],
            mark_epoch + 1,
        )
    tb.close()
