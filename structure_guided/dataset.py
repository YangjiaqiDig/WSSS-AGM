import glob
import cv2, pickle
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import pandas as pd
import random
import logging
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import torch.nn.functional as F
from options import Configs

OrgLabels = Configs().get_labels()  # ['BackGround', 'SRF', 'IRF', 'EZ', 'HRD',  'RPE']


# pd.set_option("display.max_rows", None)
logging.getLogger("PIL").setLevel(logging.WARNING)


def img_transform(
    img,
    mask,
    is_size,
    data_type,
    is_pixel_mask=False,
    layer_prob_map=None,
    uncertain_layer=None,
):
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
            if layer_prob_map is not None:
                layer_prob_map = TF.hflip(layer_prob_map)
            if uncertain_layer is not None:
                uncertain_layer = TF.hflip(uncertain_layer)

        # Random rotate
        rotate = transforms.RandomRotation(degrees=60)
        state = torch.get_rng_state()
        img = rotate(img)
        torch.set_rng_state(state)
        mask = rotate(mask)
        if layer_prob_map is not None:
            torch.set_rng_state(state)
            layer_prob_map = rotate(layer_prob_map)
        if uncertain_layer is not None:
            torch.set_rng_state(state)
            uncertain_layer = rotate(uncertain_layer)

    # to tensor
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    mask = to_tensor(mask) if not is_pixel_mask else torch.LongTensor(np.asarray(mask))

    # resize layer prob map tensor
    if layer_prob_map is not None:
        layer_prob_map = F.interpolate(
            layer_prob_map.unsqueeze(0),
            size=is_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    if uncertain_layer is not None:
        uncertain_layer = F.interpolate(
            uncertain_layer.unsqueeze(0),
            size=is_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return img, mask, layer_prob_map, uncertain_layer


def gan_normalize_transform(img, is_size):
    to_pil = transforms.ToPILImage()
    img = to_pil(img)
    resize_img = transforms.Resize(is_size)
    img = resize_img(img)
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    transform_norml = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img = transform_norml(img)
    return img


class RESCDatasetSeg(Dataset):
    def __init__(self, args, data_type, infer_list=[]):
        self.file_list = {
            "train": glob.glob("{}/train/original_images/*".format(args.root_dirs)),
            "test": glob.glob("{}/valid/original_images/*".format(args.root_dirs)),
        }
        self.labels_list = np.load(
            "{}/resc_cls_labels.npy".format(args.root_dirs), allow_pickle=True
        ).item()
        if data_type == "inference":
            self.file_list = {
                "inference": [
                    "{}/valid/original_images/{}".format(args.root_dirs, item)
                    for item in infer_list
                ]
            }
        self.data_type = data_type
        self.args = args
        # import pdb; pdb.set_trace()
        
        if data_type == "train":
            caption_dir = "/scr2/xhu/jiaqi/wsss/structure_guided/text_features/resc/blip_train_caption_f.pkl"
        else:
            caption_dir = "/scr2/xhu/jiaqi/wsss/structure_guided/text_features/resc/blip_valid_caption_f.pkl"
        with open(caption_dir, "rb") as f:
            self.caption_features = pickle.load(f)

    def __getitem__(self, idx):
        data_path = sorted(self.file_list[self.data_type])[idx]
        layer_mask_path = data_path.replace("original_images", "layer_masks").replace(
            ".bmp", ".npy"
        )
        layer_prob_map = np.load(layer_mask_path)  # (12, 224, 224)

        uncertain_layer = 1 - layer_prob_map.max(0)
        uncertain_layer[uncertain_layer <= 0.5] = 0.5
        uncertain_layer = torch.FloatTensor(uncertain_layer).unsqueeze(0)

        layer_binary_img = np.argmax(layer_prob_map, axis=0)
        # move bottom background as background
        layer_binary_img[layer_binary_img == 11] = 0

        layer_binary_img = np.uint8(layer_binary_img / layer_binary_img.max() * 255)
        layer_binary_img = np.repeat(layer_binary_img[..., np.newaxis], 3, -1)

        layer_prob_map = torch.FloatTensor(layer_prob_map)

        image_arr = np.asarray(Image.open(data_path))
        if image_arr.ndim == 2:
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)

        (
            image_tensor,
            layer_binary_tensor,
            layer_prob_tensor,
            uncertain_layer,
        ) = img_transform(
            image_arr,
            layer_binary_img,
            self.args.is_size,
            self.data_type,
            layer_prob_map=layer_prob_map,
            uncertain_layer=uncertain_layer,
        )  #

        img_name = data_path.split("/")[-1].split(".")[0]
        lesions_label = self.labels_list[img_name]
        image_label = (
            np.append([1], lesions_label)
            if "BackGround" in OrgLabels
            else lesions_label
        )
        is_lesion = 1 if 1 in lesions_label else 0
        binary_labels = torch.tensor(is_lesion)

        caption_tensor = [x for x in self.caption_features if x["path"]==data_path][0]["clip_base_embed"].cpu()
        caption_tensor.requires_grad = False
        return {
            "image": image_tensor,
            "labels": torch.from_numpy(image_label),
            "binary_labels": binary_labels,
            "path": data_path,
            "layer_prob": layer_prob_tensor,
            "layer_img": layer_binary_tensor,
            "layer_uncertain": uncertain_layer,
            "caption_tensor": caption_tensor,
        }

    def __len__(self):
        return len(self.file_list[self.data_type])


class OCTDataset(Dataset):
    def __init__(self, args, data_type, infer_list=[]):
        self.file_list = {
            "train": glob.glob("{}/original/train/*".format(args.root_dirs)),
            "test": glob.glob("{}/original/test/*".format(args.root_dirs)),
        }
        if data_type == "inference":
            self.file_list = {
                "inference": [
                    "{}/original/test/{}".format(args.root_dirs, item)
                    for item in infer_list
                ]
            }

        if data_type == "train":
            self.labels_table = pd.read_csv("/scr2/xhu/jiaqi/wsss/datasets/our_dataset/labels.csv")
        elif args.expert_annot == "mina":
            self.labels_table = pd.read_csv("/scr2/xhu/jiaqi/wsss/datasets/our_dataset/mina_labels.csv")
        elif args.expert_annot == "meera":
            self.labels_table = pd.read_csv("/scr2/xhu/jiaqi/wsss/datasets/our_dataset/meera_labels.csv")
        elif args.expert_annot == "both":
            self.labels_table = pd.read_csv("/scr2/xhu/jiaqi/wsss/datasets/our_dataset/combine_labels.csv")
        else:
            raise ValueError(
                f"No defined {args.expert_annot} annotation for NYU dataset"
            )
        if "BackGround" in OrgLabels:
            self.labels_table["BackGround"] = 1
        self.disease_img_names = self.labels_table["img"].values
        self.data_type = data_type
        self.args = args
        self.normal_label = {
            "SRF": 0,
            "IRF": 0,
            "EZ attenuated": 0,
            "EZ disrupted": 0,
            "HRD": 0,
            "RPE": 0,
            "Retinal Traction": 0,
            "Definite DRIL": 0,
            "Questionable DRIL": 0,
            "EZ": 0,
            "BackGround": 1,
        }        

    def __getitem__(self, idx):
        data_path = sorted(self.file_list[self.data_type])[idx]
        image_arr = np.asarray(Image.open(data_path))
        if image_arr.ndim == 2:
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)
        
        layer_path = data_path.replace("original", "layer_masks").split(".")[0] + ".npy"
        layer_prob_map = np.load(layer_path)

        uncertain_layer = 1 - layer_prob_map.max(0)
        uncertain_layer[uncertain_layer <= 0.5] = 0.5
        uncertain_layer = torch.FloatTensor(uncertain_layer).unsqueeze(0)

        layer_binary_img = np.argmax(layer_prob_map, axis=0)
        # move bottom background as background
        layer_binary_img[layer_binary_img == 11] = 0

        layer_binary_img = np.uint8(layer_binary_img / layer_binary_img.max() * 255)
        layer_binary_img = np.repeat(layer_binary_img[..., np.newaxis], 3, -1)

        layer_prob_map = torch.FloatTensor(layer_prob_map)

        image_name = data_path.split("/")[-1]
        (
            image_tensor,
            layer_binary_tensor,
            layer_prob_tensor,
            uncertain_layer,
        ) = img_transform(
            image_arr, 
            layer_binary_img,
            self.args.is_size,
            self.data_type,
            layer_prob_map=layer_prob_map,
            uncertain_layer=uncertain_layer,
        )
        if image_name not in self.disease_img_names:
            labels = torch.FloatTensor([self.normal_label[x] for x in OrgLabels])
        else:
            labels_df = self.labels_table.loc[self.labels_table["img"] == image_name]
            labels = torch.FloatTensor([labels_df[x].to_numpy()[0] for x in OrgLabels])
        binary_labels = torch.tensor(1 if "BackGround" in OrgLabels and labels.sum() > 1 else 0)

        return {
            "image": image_tensor,
            "labels": labels,
            "binary_labels": binary_labels,
            "path": data_path,
            "layer_prob": layer_prob_tensor,
            "layer_img": layer_binary_tensor,
            "layer_uncertain": uncertain_layer,
        }

    def __len__(self):
        return len(self.file_list[self.data_type])


class DukeDataset(Dataset):
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
        self.data_type = data_type
        self.args = args
    
    def get_mask_path_by_image_path(self, image_path):
        if "BOE" in image_path:
            mask_path = image_path.replace("/images/", "/layer_masks/")
        elif "RESC" in image_path:
            mask_path = image_path.replace("/original_images/", "/layer_masks/")
        elif "NORMAL" in image_path:
            mask_path = image_path.replace("/0.normal", "/layer_masks")
        else:
            mask_path = image_path.replace("original", "layer_masks")
        mask_path = mask_path.split(".")[0] + ".npy"
        return mask_path

    def __getitem__(self, idx):
        target_row = self.dataset_df[self.data_type].sort_values("path").iloc[idx]
        target_label = target_row["label"]
        target_path = "/scr2/xhu/jiaqi/wsss/" + target_row["path"]

        layer_path = self.get_mask_path_by_image_path(target_path)
        layer_prob_map = np.load(layer_path)

        uncertain_layer = 1 - layer_prob_map.max(0)
        uncertain_layer[uncertain_layer <= 0.5] = 0.5
        uncertain_layer = torch.FloatTensor(uncertain_layer).unsqueeze(0)

        layer_binary_img = np.argmax(layer_prob_map, axis=0)
        # move bottom background as background
        layer_binary_img[layer_binary_img == 11] = 0

        layer_binary_img = np.uint8(layer_binary_img / layer_binary_img.max() * 255)
        layer_binary_img = np.repeat(layer_binary_img[..., np.newaxis], 3, -1)

        layer_prob_map = torch.FloatTensor(layer_prob_map)

        image_arr = np.asarray(Image.open(target_path))
        if image_arr.ndim == 2:
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)
        
        (
            image_tensor,
            layer_binary_tensor,
            layer_prob_tensor,
            uncertain_layer,
        ) = img_transform(
            image_arr, 
            layer_binary_img,
            self.args.is_size,
            self.data_type,
            layer_prob_map=layer_prob_map,
            uncertain_layer=uncertain_layer,
        )
        return {
            "image": image_tensor,
            "labels": torch.FloatTensor([1, target_label]),
            "binary_labels": torch.tensor(target_label),
            "path": target_path,
            "layer_prob": layer_prob_tensor,
            "layer_img": layer_binary_tensor,
            "layer_uncertain": uncertain_layer,
        }

    def __len__(self):
        return len(self.dataset_df[self.data_type])


"""
RESC
Train: {'srf': 2934, 'ped': 402, 'retinal': 4664, 'health': 4296}
Test: {'srf': 360, 'ped': 41, 'retinal': 834, 'health': 1086}
"""