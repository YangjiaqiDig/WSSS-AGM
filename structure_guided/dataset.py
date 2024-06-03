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
from utils.utils import OrgLabels

# pd.set_option("display.max_rows", None)
logging.getLogger("PIL").setLevel(logging.WARNING)


def rotate_with_fix_state(img, rotate_f, state):
    torch.set_rng_state(state)
    img = rotate_f(img)
    return img


def img_transform(
    img,
    binarized_mask_list,
    is_size,
    data_type,
    is_pixel_mask=False,
    layer_prob_map=None,
    uncertain_layer=None,
):
    # to PIL
    to_pil = transforms.ToPILImage()
    img, binarized_mask_list = to_pil(img), [to_pil(x) for x in binarized_mask_list]

    # Resize
    resize_img = transforms.Resize(is_size)
    resize_mask = transforms.Resize(
        is_size, interpolation=transforms.InterpolationMode.NEAREST
    )  # Image.NEAREST)
    img, binarized_mask_list = resize_img(img), [
        resize_mask(x) for x in binarized_mask_list
    ]

    if data_type == "train":
        # Random color for image
        color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )
        img = color_jitter(img)

        # Random flip
        if random.random() > 0.5:
            img, binarized_mask_list = TF.hflip(img), [
                TF.hflip(x) for x in binarized_mask_list
            ]
            if layer_prob_map is not None:
                layer_prob_map = TF.hflip(layer_prob_map)
            if uncertain_layer is not None:
                uncertain_layer = TF.hflip(uncertain_layer)

        # Random rotate
        rotate = transforms.RandomRotation(degrees=60)
        state = torch.get_rng_state()
        img = rotate(img)
        binarized_mask_list = [
            rotate_with_fix_state(x, rotate, state) for x in binarized_mask_list
        ]
        if layer_prob_map is not None:
            layer_prob_map = rotate_with_fix_state(layer_prob_map, rotate, state)
        if uncertain_layer is not None:
            uncertain_layer = rotate_with_fix_state(uncertain_layer, rotate, state)

    # to tensor
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)
    binarized_mask_list = (
        [to_tensor(x) for x in binarized_mask_list]
        if not is_pixel_mask
        else [torch.LongTensor(np.asarray(x)) for x in binarized_mask_list]
    )

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

    return img, binarized_mask_list, layer_prob_map, uncertain_layer


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

        if data_type == "train":
            caption_dir = "text_features/resc/train_caption_f.pkl"
        else:
            caption_dir = "text_features/resc/valid_caption_f.pkl"
        with open(caption_dir, "rb") as f:
            self.caption_features = pickle.load(f)
        self.caption_model_v = args.caption_version.split("_")[0]
        self.caption_encode_v = "_".join(args.caption_version.split("_")[1:])

    def __getitem__(self, idx):
        data_path = sorted(self.file_list[self.data_type])[idx]
        # layer
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

        # healthy version
        healthy_path = data_path.replace("original_images", "gan_healthy").replace(
            ".bmp", ".png"
        )
        healthy_img = np.asarray(Image.open(healthy_path))

        # original image
        image_arr = np.asarray(Image.open(data_path))
        if image_arr.ndim == 2:
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)

        # transform
        (
            image_tensor,
            binarized_mask_list,
            layer_prob_tensor,
            uncertain_layer,
        ) = img_transform(
            image_arr,
            [layer_binary_img, healthy_img],
            self.args.is_size,
            self.data_type,
            layer_prob_map=layer_prob_map,
            uncertain_layer=uncertain_layer,
        )
        layer_binary_tensor = binarized_mask_list[0]
        healthy_tensor = binarized_mask_list[1]
        anomaly_diff = torch.abs(image_tensor - healthy_tensor)

        img_name = data_path.split("/")[-1].split(".")[0]
        # labels
        lesions_label = self.labels_list[img_name]
        image_label = (
            np.append([1], lesions_label)
            if "BackGround" in OrgLabels
            else lesions_label
        )
        is_lesion = 1 if 1 in lesions_label else 0
        binary_labels = torch.tensor(is_lesion)

        # caption feature
        caption_tensor = [x for x in self.caption_features if x["path"] == data_path][
            0
        ][self.caption_model_v][self.caption_encode_v].cpu()
        caption_tensor.requires_grad = False

        return {
            "path": data_path,
            "image": image_tensor,
            "labels": torch.from_numpy(image_label),
            "binary_labels": binary_labels,
            "layer_prob": layer_prob_tensor,
            "layer_img": layer_binary_tensor,
            "layer_uncertain": uncertain_layer,
            "anomaly_diff": anomaly_diff,
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
            self.labels_table = pd.read_csv("../datasets/our_dataset/labels.csv")
        elif args.expert_annot == "mina":
            self.labels_table = pd.read_csv("../datasets/our_dataset/mina_labels.csv")
        elif args.expert_annot == "meera":
            self.labels_table = pd.read_csv("../datasets/our_dataset/meera_labels.csv")
        elif args.expert_annot == "both":
            self.labels_table = pd.read_csv(
                "../datasets/our_dataset/combine_labels.csv"
            )
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

        if data_type == "train":
            caption_dir = "text_features/ours/train_caption_f.pkl"
        else:
            caption_dir = "text_features/ours/valid_caption_f.pkl"
        with open(caption_dir, "rb") as f:
            self.caption_features = pickle.load(f)

        self.caption_model_v = args.caption_version.split("_")[0]
        self.caption_encode_v = "_".join(args.caption_version.split("_")[1:])

    def __getitem__(self, idx):
        data_path = sorted(self.file_list[self.data_type])[idx]
        # original image
        image_arr = np.asarray(Image.open(data_path))
        if image_arr.ndim == 2:
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)

        layer_path = ".".join(data_path.replace("original", "layer_masks").split(".")[:-1]) + ".npy"
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

        # healthy version
        healthy_path = ".".join(data_path.replace("original", "gan_healthy").split(".")[:-1]) + ".png"
        healthy_img = np.asarray(Image.open(healthy_path))

        image_name = data_path.split("/")[-1]
        (
            image_tensor,
            binarized_mask_list,
            layer_prob_tensor,
            uncertain_layer,
        ) = img_transform(
            image_arr,
            [layer_binary_img, healthy_img],
            self.args.is_size,
            self.data_type,
            layer_prob_map=layer_prob_map,
            uncertain_layer=uncertain_layer,
        )
        layer_binary_tensor = binarized_mask_list[0]
        healthy_tensor = binarized_mask_list[1]
        anomaly_diff = torch.abs(image_tensor - healthy_tensor)

        # labels
        if image_name not in self.disease_img_names:
            labels = torch.FloatTensor([self.normal_label[x] for x in OrgLabels])
        else:
            labels_df = self.labels_table.loc[self.labels_table["img"] == image_name]
            labels = torch.FloatTensor([labels_df[x].to_numpy()[0] for x in OrgLabels])
        binary_labels = torch.tensor(
            1 if "BackGround" in OrgLabels and labels.sum() > 1 else 0
        )

        # caption feature
        caption_tensor = [x for x in self.caption_features if x["path"] == data_path][
            0
        ][self.caption_model_v][self.caption_encode_v].cpu()
        caption_tensor.requires_grad = False


        return {
            "path": data_path,
            "image": image_tensor,
            "labels": labels,
            "binary_labels": binary_labels,
            "layer_prob": layer_prob_tensor,
            "layer_img": layer_binary_tensor,
            "layer_uncertain": uncertain_layer,
            "anomaly_diff": anomaly_diff,
            "caption_tensor": caption_tensor,
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

        if data_type == "train":
            caption_dir = "text_features/duke/train_caption_f.pkl"
        else:
            caption_dir = "text_features/duke/valid_caption_f.pkl"
        with open(caption_dir, "rb") as f:
            self.caption_features = pickle.load(f)

        self.caption_model_v = args.caption_version.split("_")[0]
        self.caption_encode_v = "_".join(args.caption_version.split("_")[1:])

    def get_layer_path_by_image_path(self, image_path):
        if "BOE" in image_path:
            mask_path = image_path.replace("/images/", "/layer_masks/")
        elif "RESC" in image_path:
            mask_path = image_path.replace("/original_images/", "/layer_masks/")
        elif "NORMAL" in image_path:
            mask_path = image_path.replace("/0.normal", "/layer_masks")
        else:
            mask_path = image_path.replace("original", "layer_masks")
        mask_path = ".".join(mask_path.split(".")[:-1]) + ".npy"
        return mask_path

    def get_gan_path_by_image_path(self, image_path):
        if "BOE" in image_path:
            mask_path = image_path.replace("/images/", "/gan_healthy/")
        elif "RESC" in image_path:
            mask_path = image_path.replace("/original_images/", "/gan_healthy/")
        elif "NORMAL" in image_path:
            mask_path = image_path.replace("train/0.normal", "train/gan_healthy")
        else:
            mask_path = image_path.replace("original", "gan_healthy")
        mask_path = ".".join(mask_path.split(".")[:-1]) + ".png"
        return mask_path

    def __getitem__(self, idx):
        target_row = self.dataset_df[self.data_type].sort_values("path").iloc[idx]
        target_label = target_row["label"]
        target_path = "../" + target_row["path"]

        layer_path = self.get_layer_path_by_image_path(target_path)
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

        # healthy version
        healthy_path = self.get_gan_path_by_image_path(target_path)
        healthy_img = np.asarray(Image.open(healthy_path))

        # original image
        image_arr = np.asarray(Image.open(target_path))
        if image_arr.ndim == 2:
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)

        (
            image_tensor,
            binarized_mask_list,
            layer_prob_tensor,
            uncertain_layer,
        ) = img_transform(
            image_arr,
            [layer_binary_img, healthy_img],
            self.args.is_size,
            self.data_type,
            layer_prob_map=layer_prob_map,
            uncertain_layer=uncertain_layer,
        )
        layer_binary_tensor = binarized_mask_list[0]
        healthy_tensor = binarized_mask_list[1]
        anomaly_diff = torch.abs(image_tensor - healthy_tensor)

        # caption feature
        caption_tensor = [x for x in self.caption_features if x["path"] == target_path][
            0
        ][self.caption_model_v][self.caption_encode_v].cpu()
        caption_tensor.requires_grad = False

        return {
            "path": target_path,
            "image": image_tensor,
            "labels": torch.FloatTensor([1, target_label]),
            "binary_labels": torch.tensor(target_label),
            "layer_prob": layer_prob_tensor,
            "layer_img": layer_binary_tensor,
            "layer_uncertain": uncertain_layer,
            "anomaly_diff": anomaly_diff,
            "caption_tensor": caption_tensor,
        }

    def __len__(self):
        return len(self.dataset_df[self.data_type])


"""
RESC
Train: {'srf': 2934, 'ped': 402, 'retinal': 4664, 'health': 4296}
Test: {'srf': 360, 'ped': 41, 'retinal': 834, 'health': 1086}
"""
