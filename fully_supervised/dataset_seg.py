import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.utils_full_seg import (
    convert_data_labels,
)
import torch
import random
import logging
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import pandas as pd

# pd.set_option("display.max_rows", None)
logging.getLogger("PIL").setLevel(logging.WARNING)


def img_transform(img, mask, is_size, data_type, is_pixel_mask=False, layer_mask=None):
    # to PIL
    to_pil = transforms.ToPILImage()
    img, mask = to_pil(img), to_pil(mask)

    # Resize
    resize_img = transforms.Resize(is_size)
    resize_mask = transforms.Resize(
        is_size, interpolation=transforms.InterpolationMode.NEAREST
    )  # Image.NEAREST)
    img, mask = resize_img(img), resize_mask(mask)

    if data_type == "train":
        # Random color for image
        color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )
        img = color_jitter(img)

        # Random flip
        if random.random() > 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
            if layer_mask is not None:
                layer_mask = TF.hflip(layer_mask)

        # Random rotate
        rotate = transforms.RandomRotation(degrees=60)
        state = torch.get_rng_state()
        img = rotate(img)
        torch.set_rng_state(state)
        mask = rotate(mask)
        if layer_mask is not None:
            torch.set_rng_state(state)
            layer_mask = rotate(layer_mask)

    # to tensor
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    mask = to_tensor(mask) if not is_pixel_mask else torch.LongTensor(np.array(mask))

    return img, mask, layer_mask


class RESCDatasetSeg(Dataset):
    def __init__(self, args, data_type, infer_list=[]):
        self.file_list = {
            "train": glob.glob("{}/train/original_images/*".format(args.root_dirs)),
            "test": glob.glob("{}/valid/original_images/*".format(args.root_dirs)),
        }
        self.label_list = {
            # "train": glob.glob("{}/train/label_images/*".format(args.root_dirs)),
            "train": glob.glob("/data2/xiaolhu/jiaqi/baseline_models/TPRO/work_dirs/resc/pseudo_labels/*"),
            "test": glob.glob("{}/valid/label_images/*".format(args.root_dirs)),
        }
        if data_type == "inference":
            self.file_list = {
                "inference": [
                    "{}/valid/original_images/{}".format(args.root_dirs, item)
                    for item in infer_list
                ]
            }
            self.label_list = {
                "inference": [
                    "{}/valid/label_images/{}".format(args.root_dirs, item)
                    for item in infer_list
                ]
            }
        img_level_labels = np.load(
            "{}/resc_cls_labels.npy".format(args.root_dirs), allow_pickle=True
        ).item()
        non_black_img = [
            key for key, value in img_level_labels.items() if value.sum() > 0
        ]
        self.file_list["train"] = [
            item
            for item in self.file_list["train"]
            if item.split("/")[-1].split(".")[0] in non_black_img
        ]
        self.label_list["train"] = [
            item
            for item in self.label_list["train"]
            if item.split("/")[-1].split(".")[0] in non_black_img
        ]
        self.file_list["test"] = [
            item
            for item in self.file_list["test"]
            if item.split("/")[-1].split(".")[0] in non_black_img
        ]
        self.label_list["test"] = [
            item
            for item in self.label_list["test"]
            if item.split("/")[-1].split(".")[0] in non_black_img
        ]
        # import pdb; pdb.set_trace()
        self.data_type = data_type
        self.args = args

    def __getitem__(self, idx):
        data_path = sorted(self.file_list[self.data_type])[idx]
        label_path = sorted(self.label_list[self.data_type])[idx]
        layer_mask_path = data_path.replace("original_images", "layer_masks").replace(
            ".bmp", ".npy"
        )
        layer_prob_map = np.load(layer_mask_path)  # (12, 244, 244)
        layer_prob_map = torch.FloatTensor(layer_prob_map)
        layer_prob_map = F.interpolate(
            layer_prob_map.unsqueeze(0),
            size=self.args.is_size,
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        image_arr = np.asarray(Image.open(data_path))
        if image_arr.ndim == 2:
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)

        if "npy" in label_path:
            orig_annot = np.load(label_path)
            bg_score = np.ones((1, orig_annot.shape[1], orig_annot.shape[2])) * 0.7
            orig_annot = np.argmax(np.concatenate((bg_score, orig_annot), axis=0), axis=0).astype(np.uint8)
        else:
            orig_annot = np.asarray(Image.open(label_path))
        if orig_annot.ndim == 2:
            orig_annot = np.repeat(orig_annot[..., np.newaxis], 3, -1)
        if "npy" in label_path:
            pixel_mask = orig_annot
        else:
            pixel_mask = convert_data_labels(orig_annot, self.args.root_dirs)
        image_tensor, mask_tensor, layer_mask_tensor = img_transform(
            image_arr,
            pixel_mask,
            self.args.is_size,
            self.data_type,
            is_pixel_mask=True,
            layer_mask=layer_prob_map,
        )  #
        mask_tensor = mask_tensor[..., 0]

        return {
            "image": image_tensor,
            "labels": mask_tensor,
            "path": data_path,
            "layer_mask": layer_mask_tensor,
        }

    def __len__(self):
        return len(self.file_list[self.data_type])


class DukeDatasetSeg(Dataset):
    def __init__(self, args, data_type, infer_list=[]):
        self.dataset_df = {
            "train": pd.read_csv("{}/train.csv".format(args.root_dirs), index_col=0),
            "test": pd.read_csv("{}/valid.csv".format(args.root_dirs), index_col=0),
        }
        if data_type == "inference":
            val_df = pd.read_csv("{}/valid.csv".format(args.root_dirs), index_col=0)
            if infer_list:
                pat = "|".join(r"\b{}\b".format(x) for x in infer_list)
                val_df = val_df[val_df["path"].str.contains(pat)]
            self.dataset_df = {"inference": val_df}

        self.dataset_df["train"] = self.dataset_df["train"][
            self.dataset_df["train"]["label"]==1
        ]
        self.dataset_df["test"] = self.dataset_df["test"][
            self.dataset_df["test"]["label"]==1
        ]
        # import pdb; pdb.set_trace()
        self.data_type = data_type
        self.args = args
        self.train_label_dir = "/data2/xiaolhu/jiaqi/baseline_models/TPRO/work_dirs/duke/pseudo_labels/"

    def __getitem__(self, idx):
        target_row = self.dataset_df[self.data_type].sort_values("path").iloc[idx]
        target_path = "../" + target_row["path"]
        if self.data_type == "train":
            pixel_label_path = self.train_label_dir + target_row["path"].replace("/", "+").split(".")[0]+".npy"
        else:
            pixel_label_path = target_path.replace("images", "labels")
    
        image_arr = np.asarray(Image.open(target_path))
        if image_arr.ndim == 2:
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)

        if "npy" in pixel_label_path:
            orig_annot = np.load(pixel_label_path)
            bg_score = np.ones((1, orig_annot.shape[1], orig_annot.shape[2])) * 0.78 # duke val best 0.77
            orig_annot = np.argmax(np.concatenate((bg_score, orig_annot), axis=0), axis=0).astype(np.uint8)
        else:
            orig_annot = np.asarray(Image.open(pixel_label_path))
        if orig_annot.ndim == 2:
            orig_annot = np.repeat(orig_annot[..., np.newaxis], 3, -1)
        if "npy" in pixel_label_path:
            pixel_mask = orig_annot
        else:
            pixel_mask = convert_data_labels(orig_annot, self.args.root_dirs)
        image_tensor, mask_tensor,_ = img_transform(
            image_arr,
            pixel_mask,
            self.args.is_size,
            self.data_type,
            is_pixel_mask=True,
        )  #
        mask_tensor = mask_tensor[..., 0]

        return {
            "image": image_tensor,
            "labels": mask_tensor,
            "path": target_path,
            "layer_mask": mask_tensor,
        }

    def __len__(self):
        return len(self.dataset_df[self.data_type])
