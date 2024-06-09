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

type_color = {
    0: [0, 0, 0],  # black
    1: [52, 195, 235],  # srf red
    3: [0, 255, 0],  # irf green
    2: [165, 235, 52],  # ez blue   235, 211, 52
    4: [255, 255, 0],  # hrd yellow
    5: [255, 165, 0],  # rpe orange
    6: [255, 0, 255],  # back ground pink
}
OrgLabels = Configs().get_labels()  # ['BackGround', 'SRF', 'IRF', 'EZ', 'HRD',  'RPE']


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


def load_mask(image_path):
    orig_mask = np.asarray(Image.open(get_mask_path_by_image_path(image_path)))[..., 0]
    orig_mask[orig_mask > 150] = 255
    orig_mask[orig_mask <= 150] = 0
    return orig_mask


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
    # norm_cams[norm_cams < cam_min + 1e-5] = 0

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


def get_cam_and_save(
    cam_preds, input_data, opts, epoch, orig_cams, layer_cams
):
    binary_labels = input_data["binary_labels"]
    labels = input_data["labels"]
    path = input_data["path"]
    # batch_preds [BC] B: batch, C: Class
    ready_cam_pred = []
    gt = []
    for batch_nb, cam_pred in enumerate(cam_preds):
        # only calculate and save for ground truth lesion images
        if binary_labels[batch_nb] == 0:
            continue
        image_path = path[batch_nb]
        img_name = image_path.split("/")[-1]
        # 0-255 pixel value, numpy
        orig_img = np.asarray(Image.open(image_path))
        if len(orig_img.shape) == 3:
            orig_img = orig_img[..., 0]

        orig_annot = get_annot_by_dataset(opts, img_name)
        truth_label = [
            OrgLabels[idx]
            for idx, l in enumerate(labels[batch_nb])
            if l != 0 and OrgLabels[idx] != "BackGround"
        ]
        truth_label = "_".join(truth_label)

        resize_cam = F.interpolate(
            cam_pred.clone().unsqueeze(0),  # [1, cls, w, h]
            size=(orig_img.shape[0], orig_img.shape[1]),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

        binarized_seg = resize_cam.argmax(0)
        ready_cam_pred.append(binarized_seg.detach().cpu().numpy())

        gt_labels = convert_data_labels(orig_annot, opts.root_dirs)
        gt.append(gt_labels)

        if not opts.save_results:
            continue

        orig_cam = orig_cams[batch_nb]
        layer_cam = layer_cams[batch_nb]

        save_cam_in_row = [input_data["image"][batch_nb][0]]  # orig
        # import pdb; pdb.set_trace()
        for i, name in enumerate(OrgLabels):
            if name == "BackGround" or labels[batch_nb][i] == 0:
                continue
            save_cam_in_row.append(cam_pred[i].cpu())  # comb_cam
            save_cam_in_row.append(
                torch.tensor(orig_cam[i - 1])
            )  # background is 0, but we dont have background cam
        # save_cam_in_row.append(
        #     torch.tensor(layer_cam[0])
        # )  # we only have 1 layer of lesion cam
        # import pdb; pdb.set_trace()
        save_cam_in_row = torch.stack(save_cam_in_row).unsqueeze(1)

        save_seg = (
            torch.clamp(binarized_seg / get_num_lesions(), 0, 1)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )
        save_seg = np.uint8(save_seg * 255)
        layer_seg_result = input_data["layer_prob"][batch_nb]
        resized_layer = F.interpolate(
            layer_seg_result.unsqueeze(0),  # [1, cls, w, h]
            size=(orig_img.shape[0], orig_img.shape[1]),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        resized_layer = resized_layer.argmax(0).detach().cpu().numpy()
        resized_layer = np.uint8(resized_layer / resized_layer.max() * 255)

        save_img = [orig_img, orig_annot, save_seg, resized_layer]
        save_image_h = cv2.hconcat(save_img)

        save_path = os.path.join(
            opts.save_folder, "images", "{}".format(img_name.split(".")[0])
        )
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        is_seg_ext = "cam"
        cv2.imwrite(
            save_path + "/epoch{}_{}_{}.jpg".format(epoch, truth_label, is_seg_ext),
            save_image_h,
        )
        vutils.save_image(
            save_cam_in_row,
            save_path + "/epoch{}_{}_heatmaps.jpg".format(epoch, truth_label),
            scale_each=True,
            normalize=True,
        )
    return gt, ready_cam_pred

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
        k: get_target_cams_tensor(k, v, multi_labels, binary_labels) for k, v in cams_dict.items()
    }

    updated_cams_dicts = []
    for batch_nb in range(len(binary_labels)):
        # only calculate and save for ground truth lesion images
        if binary_labels[batch_nb] == 0:
            continue
        img_name = path[batch_nb].split("/")[-1]
        # 0-255 pixel value, numpy
        orig_annot = get_annot_by_dataset(opts, img_name)
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
        # updated_single_cam["final_cam"] = (
        #     0.4 * updated_single_cam["clip-l3-relu-sim"]
        #     + 0.6 * updated_single_cam["clip-l4-relu-sim"]
        # )
        if "clip-l4-relu-sim" not in updated_single_cam:
            updated_single_cam["final_cam"] = updated_single_cam["main-relu-cam"]
        else:
            updated_single_cam["final_cam"] = (
                0.3 * updated_single_cam["clip-l3-relu-sim"]
                + 0.3 * updated_single_cam["clip-l4-relu-sim"]
                + 0.4 * updated_single_cam["main-relu-cam"]
            )

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
