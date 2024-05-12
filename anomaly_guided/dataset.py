import glob
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms as T
from utils.utils import OrgLabels, convert_resc_labels, get_mask_path_by_image_path
import torch
import pandas as pd
import random
import logging
import torchvision.transforms.functional as TF
import torchvision.utils as vutils


# pd.set_option("display.max_rows", None)
logging.getLogger('PIL').setLevel(logging.WARNING)

def img_transform(img, mask, is_size, data_type, is_seg=False, healthy_v=None):
    # to PIL
    to_pil = T.ToPILImage()
    img, mask = to_pil(img), to_pil(mask)
    
    # Resize 
    resize_img = T.Resize(is_size)
    resize_mask = T.Resize(is_size, interpolation=T.InterpolationMode.NEAREST)
    img, mask = resize_img(img), resize_mask(mask)
    
    if data_type == 'train':
        # Random color for image
        color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        img = color_jitter(img)

        # Random flip
        if random.random() > 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        
        # Random rotate
        rotate = T.RandomRotation(degrees=60)
        state = torch.get_rng_state()
        img = rotate(img)
        torch.set_rng_state(state)
        mask = rotate(mask)

    # to tensor
    to_tensor = T.ToTensor()
    if not is_seg:
        mask = to_tensor(mask)
    else:
        mask = torch.LongTensor(np.asarray(mask))
    img = to_tensor(img)
    
    return img, mask

def gan_normalize_transform(img, is_size):
    to_pil = T.ToPILImage()
    img = to_pil(img)
    resize_img = T.Resize(is_size)
    img = resize_img(img)
    to_tensor = T.ToTensor()
    img = to_tensor(img)
    transform_norml = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img = transform_norml(img)
    return img

class OCTDataset(Dataset): 
    def __init__(self, args, data_type, infer_list=[]):
        self.file_list = {'train': glob.glob("{}/original/train/*".format(args.root_dirs)), 'test': glob.glob("{}/original/test/*".format(args.root_dirs))}
        self.mask_list = {'train': glob.glob("{}/train/*.png".format(args.mask_dir)), 'test': glob.glob("{}/test/*.png".format(args.mask_dir))}
        if data_type == 'inference':
            self.file_list = {'inference': ["{}/original/test/{}".format(args.root_dirs, item) for item in infer_list]}
            self.mask_list = {'inference': ["{}/test/{}.png".format(args.mask_dir, item.split('.')[0]) for item in infer_list]}
        
        if data_type == 'train':
            self.labels_table = pd.read_csv("datasets/our_dataset/labels.csv")
        elif args.expert_annot == 'mina':
            self.labels_table = pd.read_csv("datasets/our_dataset/mina_labels.csv")
        elif args.expert_annot == 'meera':
            self.labels_table = pd.read_csv("datasets/our_dataset/meera_labels.csv")
        elif args.expert_annot == 'both':
            self.labels_table = pd.read_csv("datasets/our_dataset/combine_labels.csv")
        else:
            raise ValueError(f"No defined {args.expert_annot} annotation for NYU dataset")
            
        if 'BackGround' in OrgLabels:
            self.labels_table['BackGround'] = 1
        self.disease_img_names = self.labels_table['img'].values
        self.data_type = data_type
        self.args = args
        self.normal_label = {'SRF': 0, 'IRF': 0, 'EZ attenuated': 0,  'EZ disrupted': 0,  'HRD': 0,  'RPE': 0,  'Retinal Traction': 0,  'Definite DRIL': 0,  'Questionable DRIL': 0,  'EZ': 0,  'BackGround': 1}

    def __getitem__(self, idx):
        data_path = sorted(self.file_list[self.data_type])[idx]
        mask_path = sorted(self.mask_list[self.data_type])[idx]
        image = Image.open(data_path)
        mask = np.asarray(Image.open(mask_path))
        image_arr = np.asarray(image)
        if (image_arr.ndim == 2):
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)
        image_name = data_path.split('/')[-1]
        image_tensor, mask_tensor = img_transform(image_arr, mask, self.args.is_size, self.data_type)
        if image_name not in self.disease_img_names:
            labels = torch.FloatTensor([self.normal_label[x] for x in OrgLabels])
        else:
            labels_df = self.labels_table.loc[self.labels_table['img'] == image_name]
            labels = torch.FloatTensor([labels_df[x].to_numpy()[0] for x in OrgLabels])
        return {'image': image_tensor, 'labels': labels, 'path': data_path, 'mask': mask_tensor, 'shape': image_arr.shape, 'mask_path': mask_path}
    
    def __len__(self):
        return len(self.file_list[self.data_type])
    
class RESCDataset(Dataset):
    def __init__(self, args, data_type, infer_list=[], is_generate_pseudo_label=False):
        self.file_list = {'train': glob.glob("{}/train/original_images/*".format(args.root_dirs)), 'test': glob.glob("{}/valid/original_images/*".format(args.root_dirs))}
        self.mask_list = {'train': glob.glob("{}/train/*".format(args.mask_dir)), 'test': glob.glob("{}/valid/*".format(args.mask_dir))}
        self.labels_list = np.load('{}/resc_cls_labels.npy'.format(args.root_dirs), allow_pickle=True).item()
        if data_type == 'inference':
            self.file_list = {'inference': ["{}/valid/original_images/{}".format(args.root_dirs, item) for item in infer_list]}
            self.mask_list = {'inference': ["{}/valid/{}".format(args.mask_dir, item) for item in infer_list]}
        self.data_type = data_type
        self.args = args
        self.is_generate_pseudo_label = is_generate_pseudo_label
        # self.gan_pretrained = gan_pretrained

    def __getitem__(self, idx):
        data_path = sorted(self.file_list[self.data_type])[idx]
        mask_path = sorted(self.mask_list[self.data_type])[idx]
        
        image = Image.open(data_path)
        mask = np.asarray(Image.open(mask_path))
        image_arr = np.asarray(image)        
        if (image_arr.ndim == 2):
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)
        transform_type = 'test' if self.is_generate_pseudo_label else self.data_type
        # with torch.no_grad():
        #     org_img = gan_normalize_transform(image_arr, self.args.is_size)
        #     healthy_v = self.gan_pretrained.inference(org_img.unsqueeze(0)).squeeze(0)
        #     vutils.save_image(healthy_v, 'healthy_v.png', normalize=True)
        # import pdb; pdb.set_trace()
        image_tensor, mask_tensor= img_transform(image_arr, mask, self.args.is_size, transform_type)
        
        img_name = data_path.split('/')[-1].split('.')[0]
        image_label = np.append(self.labels_list[img_name], [1]) if 'BackGround' in OrgLabels else self.labels_list[img_name]
        return {'image': image_tensor, 'labels': torch.from_numpy(image_label), 'path': data_path, 'mask': mask_tensor, 'shape': image_arr.shape, 'mask_path': mask_path}

    def __len__(self):
        return len(self.file_list[self.data_type])

class DukeDataset(Dataset):
    def __init__(self, args, data_type, infer_list=[], is_generate_pseudo_label=False):
        self.dataset_df = {'train': pd.read_csv('{}/train.csv'.format(args.root_dirs), index_col=0), 'test': pd.read_csv('{}/valid.csv'.format(args.root_dirs), index_col=0)}
        if data_type == 'inference':
            val_df = pd.read_csv("{}/valid.csv".format(args.root_dirs), index_col=0)
            if infer_list:
                pat = '|'.join(r'\b{}\b'.format(x) for x in infer_list)
                val_df = val_df[val_df['path'].str.contains(pat)]
            self.dataset_df = {'inference': val_df}
        self.data_type = data_type
        self.args = args
        self.is_generate_pseudo_label = is_generate_pseudo_label

    def __getitem__(self, idx):
        target_row = self.dataset_df[self.data_type].sort_values('path').iloc[idx]
        target_label = target_row['label']
        target_path = target_row['path']
        mask_path = get_mask_path_by_image_path(target_path)
        image = Image.open(target_path)
        mask = np.asarray(Image.open(mask_path))
        mask[mask>150]=255
        mask[mask<=150]=0

        image_arr = np.asarray(image)
        if (image_arr.ndim == 2):
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)
        transform_type = 'test' if self.is_generate_pseudo_label else self.data_type
        image_tensor, mask_tensor = img_transform(image_arr, mask, self.args.is_size, transform_type)
        return {'image': image_tensor, 'labels': torch.FloatTensor([target_label, 1]), 'path': target_path, 'mask': mask_tensor, 'shape': image_arr.shape, 'mask_path': mask_path}
    def __len__(self):
        return len(self.dataset_df[self.data_type])


class SegmentRESCDataset(Dataset):
    def __init__(self, args, data_type, infer_list=[]):
        self.labels_list = {'train': glob.glob("{}/pseudo_label/*".format(args.root_dirs))}
        save_labels = np.load(f'{args.root_dirs}/resc_cls_labels.npy', allow_pickle=True).item()
        test_label_paths = glob.glob("{}/valid/label_images/*".format(args.root_dirs))
        seg_label_paths = [x for x in test_label_paths if save_labels[x.split('/')[-1].split('.')[0]].sum() > 0]
        self.labels_list['test'] = seg_label_paths
        if data_type == 'inference':
            self.labels_list = {'inference': ["{}/valid/label_images/{}".format(args.mask_dir, item) for item in infer_list]}
        self.data_type = data_type
        self.args = args

    def __getitem__(self, idx):
        data_path = sorted(self.labels_list[self.data_type])[idx]
        img_name = data_path.split('/')[-1].split('.')[0]
        d_path = 'train' if self.data_type == 'train' else 'valid'
        orig_img_path = f"{self.args.root_dirs}/{d_path}/original_images/{img_name}.bmp"
        image = np.asarray(Image.open(orig_img_path))
        annot = np.asarray(Image.open(data_path))
        if self.data_type != 'train':
            annot = convert_resc_labels(annot)
        if (image.ndim == 2):
            image = np.repeat(image[..., np.newaxis], 3, -1)
        image_tensor, annot_tensor = img_transform(image, annot, self.args.is_size, self.data_type, is_seg=True)
        # import pdb; pdb.set_trace()
        return {'image': image_tensor, 'labels': annot_tensor, 'path': data_path}

    def __len__(self):
        return len(self.labels_list[self.data_type])

"""
Author: Manpreet Singh Minhas
Contact: msminhas at uwaterloo ca
"""
from pathlib import Path
from typing import Any, Callable, Optional

from torchvision.datasets.vision import VisionDataset


class SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale") -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            mask_folder (str): Name of the folder that contains the masks in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.
        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == "Train":
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[
                    int(np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[
                    int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path,
                                                        "rb") as mask_file:
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")
            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")
            sample = {"image": image, "mask": mask}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])
            return sample

if __name__ == "__main__":
    '''
    RESC
    Train: {'srf': 2934, 'ped': 402, 'retinal': 4664, 'health': 4296}
    Test: {'srf': 360, 'ped': 41, 'retinal': 834, 'health': 1086}
    '''
    class Args():
        def __init__(self) -> None:
            self.root_dirs = 'datasets/2015_BOE_Chiu'
            self.mask_dir = 'datasets/RESC/mask'
            self.is_size = (512, 512)
    args = Args()
    target_dataset = DukeDataset(args, 'test')
    # l = {'srf': 0, 'ped': 0, 'retinal': 0, 'health': 0}
    for idx in range(0, len(target_dataset)):
        res = target_dataset[idx]
        print(res)
        import pdb; pdb.set_trace()
    
