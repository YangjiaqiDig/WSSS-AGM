import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils import OrgLabels
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import os
# pd.set_option("display.max_rows", None)

class OCTDataset(Dataset): 
    def __init__(self, args, transform, data_type):
        self.file_list = {'train': glob.glob("{}/train/*".format(args.root_dirs)), 'test': glob.glob("{}/test/*".format(args.root_dirs))}
        self.labels_table = pd.read_csv("our_dataset/labels.csv")
        if args.combine_ez:
            self.labels_table['EZ'] = self.labels_table['EZ attenuated'] + self.labels_table['EZ disrupted']
            self.labels_table.loc[self.labels_table['EZ'] > 1, 'EZ'] = 1
        if OrgLabels[-1] == 'BackGround':
            self.labels_table['BackGround'] = 1
        self.transform = transform
        self.roots = args.root_dirs
        self.input_structure = args.input_structure
        self.data_type = data_type

    def __getitem__(self, idx):
        data_path = sorted(self.file_list[self.data_type])[idx]
        image = Image.open(data_path)
        image_arr = np.asarray(image) 
        if (image_arr.ndim == 2):
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)
        image_name = data_path.split('/')[-1]
        image_tensor = self.transform(image_arr)
        if image_name in self.labels_table['img'].values:
            labels = self.labels_table.loc[self.labels_table['img'] == image_name]
        else:
            normal_label = {'SRF': 0, 'IRF': 0, 'EZ attenuated': 0,  'EZ disrupted': 0,  'HRD': 0,  'RPE': 0,  'Retinal Traction': 0,  'Definite DRIL': 0,  'Questionable DRIL': 0,  'EZ': 0,  'BackGround': 1}
            return {'image': image_tensor, 'labels': torch.FloatTensor([normal_label[x] for x in OrgLabels]), 'path': data_path}
        
        # if self.input_structure:
        #     # not re-generate structure for background removal
        #     str_image_gan = Image.open('our_dataset/structures/gan/{}'.format(image_name))
        #     str_image_orig = Image.open('our_dataset/structures/original/{}'.format(image_name))
        #     str_tensor_gan, str_tensor_orig = self.transform_val(np.asarray(str_image_gan)), self.transform_val(np.asarray(str_image_orig))
        #     image_tensor = torch.cat((image_tensor, str_tensor_orig, str_tensor_gan))
        try:
            return {'image': image_tensor, 'labels': torch.FloatTensor([labels[x].to_numpy()[0] for x in OrgLabels]), 'path': data_path}
        except: print("???", data_path, image_tensor.shape, labels)
    def __len__(self):
        return len(self.file_list[self.data_type])
        
def train_transform(is_size):
    transform_seq = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(is_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(15)),
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

if __name__ == "__main__":
    root_dirs = ["our_dataset/original/DME_1", "our_dataset/original/DME_2", "our_dataset/original/DME_3", "our_dataset/original/DME_4", "our_dataset/original/DR"]
    labels_table = []
    for root_dir in root_dirs:
        labels_table.append(pd.read_csv("%s/labels.csv" % root_dir)) 
    labels_table = pd.concat(labels_table, ignore_index=True)
    labels_table.to_csv('our_dataset/labels.csv', index=False)
    print(labels_table)

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
        
