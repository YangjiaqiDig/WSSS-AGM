import glob

from matplotlib import image
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from oct_utils import OrgLabels
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# pd.set_option("display.max_rows", None)

class OCTDataset(Dataset): 
    def __init__(self, args, transform_train=None, transform_val=None):
        self.file_list, self.labels_table = [], []
        self.remove_background = args.remove_background
        for root_dir in args.root_dirs:
            imgs_path = 'images_backrm' if self.remove_background else 'images'
            self.file_list += glob.glob("{}/{}/*".format(root_dir, imgs_path))
            self.labels_table.append(pd.read_csv("%s/labels.csv" % root_dir)) 
        self.labels_table = pd.concat(self.labels_table, ignore_index=True)
        if args.combine_ez:
            self.labels_table['EZ'] = self.labels_table['EZ attenuated'] + self.labels_table['EZ disrupted']
            self.labels_table.loc[self.labels_table['EZ'] > 1, 'EZ'] = 1
        if OrgLabels[-1] == 'BackGround':
            self.labels_table['BackGround'] = 1
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.roots = args.root_dirs
        self.input_gan = args.input_gan
        self.input_structure = args.input_structure
    def set_use_train_transform(self):
        return
    def __getitem__(self, idx):
        data_path = sorted(self.file_list)[idx]
        image = Image.open(data_path)
        image_arr = np.asarray(image) 
        if (image_arr.ndim == 2):
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)
        t_func = transforms.ToTensor()
        image_name = data_path.split('/')[-1]
        labels = self.labels_table.loc[self.labels_table['img'] == image_name]
        if self.transform_train:
            image_tensor = self.transform(image_arr)
        else: image_tensor = t_func(image)
        if self.input_gan:
            gan_path = 'our_dataset/ganomaly_results_backrm/1.abnormal' if self.remove_background else 'our_dataset/ganomaly_results/1.abnormal'
            gan_image = Image.open('{}/{}'.format(gan_path, image_name))
            gan_tensor = t_func(gan_image)
            image_tensor = torch.cat((image_tensor, gan_tensor))
        if self.input_structure:
            # not re-generate structure for background removal
            str_image_gan = Image.open('our_dataset/structures/gan/{}'.format(image_name))
            str_image_orig = Image.open('our_dataset/structures/original/{}'.format(image_name))
            str_tensor_gan, str_tensor_orig = self.transform(np.asarray(str_image_gan)), self.transform(np.asarray(str_image_orig))
            image_tensor = torch.cat((image_tensor, str_tensor_orig, str_tensor_gan))
        try:
            return {'image': image_tensor, 'labels': torch.FloatTensor([labels[x].to_numpy()[0] for x in OrgLabels]), 'path': data_path}
        except: print("???", data_path, image_tensor.shape, labels)
    def __len__(self):
        return len(self.file_list)
        
def train_transform(is_size):
    transform_seq = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(is_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #  std=[0.229, 0.224, 0.225])
    ])
    return transform_seq

def valid_transform(is_size):
    transform_seq = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(is_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    return transform_seq

def normal_transform(is_size):
    transform_seq = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(is_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    return transform_seq

if __name__ == "__main__":
    root_dirs = ["our_dataset/dataset_DR", "our_dataset/dataset_DME/1", "our_dataset/dataset_DME/3"]
    dataset = OCTDataset(root_dirs, transform=valid_transform())
    acc, f1m, f1mi = 0, 0, 0
    gt = [0.0, 1.0, ]#0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    gt_dr = [0.0, 1.0,]# 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for data in dataset:
        if (data["path"].split('/')[0] == "dataset_DR" ): 
            gt_f = gt_dr
        else: gt_f = gt
        acc +=  accuracy_score(gt_f,data["labels"].numpy())
        f1m += f1_score(gt_f,data["labels"].numpy(),average = 'macro', zero_division=1)
        f1mi += f1_score(gt_f,data["labels"].numpy(),average = 'micro', zero_division=1)
    print(acc / len(dataset), f1m / len(dataset), f1mi / len(dataset))
        
