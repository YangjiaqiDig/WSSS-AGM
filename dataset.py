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
    def __init__(self, root_dirs, transform=None):
        self.file_list, self.labels_table = [], []
        for root_dir in root_dirs:
            self.file_list += glob.glob("%s/images/*" % root_dir)
            self.labels_table.append(pd.read_csv("%s/labels.csv" % root_dir)) 
        self.labels_table = pd.concat(self.labels_table, ignore_index=True)
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
        try:
            return {'image': image_tensor, 'labels': torch.FloatTensor([labels[x].to_numpy()[0] for x in OrgLabels]), 'path': data_path}
        except: print("???", data_path, image_tensor.shape, labels)
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
    dataset = OCTDataset(root_dirs, transform=valid_transform())
    acc, f1m, f1mi = 0, 0, 0
    gt = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    gt_dr = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for data in dataset:
        if (data["path"].split('/')[0] == "dataset_DR" ): 
            print('hey')
            gt_f = gt_dr
        else: gt_f = gt
        acc +=  accuracy_score(gt_f,data["labels"].numpy())
        f1m += f1_score(gt_f,data["labels"].numpy(),average = 'macro', zero_division=1)
        f1mi += f1_score(gt_f,data["labels"].numpy(),average = 'micro', zero_division=1)
    print(acc / len(dataset), f1m / len(dataset), f1mi / len(dataset))
        
