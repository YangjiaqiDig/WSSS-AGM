import glob

from matplotlib import image
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from oct_utils import LABELS
import torch

class OCTDataset(Dataset): 
    def __init__(self, root_dir, transform=None):
        self.file_list = glob.glob("%s/*.npy" % root_dir)
        self.transform = transform
    def __getitem__(self, idx):
        data_path = sorted(self.file_list)[idx]
        data_pack = np.load(data_path, allow_pickle=True)
        dict_data = data_pack.item()
        img = dict_data["image"]
        
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'labels': torch.FloatTensor([dict_data[x] for x in LABELS])}
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
    root_dir = "train_dr"
    dataset = OCTDataset(root_dir, transform=train_transform())
    for i in dataset:
        print(i["image"].shape)
        
