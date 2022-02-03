import glob

from matplotlib import image
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from oct_utils import OrgLabels
import torch
import pandas as pd

class OCTDataset(Dataset): 
    def __init__(self, root_dirs, transform=None):
        self.file_list, self.labels_table = [], []
        for root_dir in root_dirs:
            self.file_list += glob.glob("%s/images/*" % root_dir)
            self.labels_table.append(pd.read_csv("%s/labels.csv" % root_dir)) 
        self.labels_table = pd.concat(self.labels_table)
        self.transform = transform
        self.roots = root_dirs
    def __getitem__(self, idx):
        data_path = sorted(self.file_list)[idx]
        image = Image.open(data_path)
        image_arr = np.asarray(image) 
        if (image_arr.ndim == 2):
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)
        t_func = transforms.ToTensor()
        image_name = data_path.split('/')[-1]
        labels = self.labels_table.loc[self.labels_table['img'] == image_name]
        if self.transform:
            image_tensor = self.transform(image_arr)
        else: image_tensor = t_func(image)
        return {'image': image_tensor, 'labels': torch.FloatTensor([labels[x].to_numpy()[0] for x in OrgLabels]), 'path': data_path}
    def __len__(self):
        return len(self.file_list)
        
def train_transform():
    transform_seq = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((500,750)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #  std=[0.229, 0.224, 0.225])
    ])
    return transform_seq

def valid_transform():
    transform_seq = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((500,750)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    return transform_seq


if __name__ == "__main__":
    root_dirs = ["dataset_DR", "dataset_DME/1", "dataset_DME/3"]
    dataset = OCTDataset(root_dirs, transform=train_transform())
    print(dataset[0])
    # for i in dataset:
    #     print(i["image"].shape)
        
