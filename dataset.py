import glob
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.utils as vutils
from utils import OrgLabels
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import random
import logging
# pd.set_option("display.max_rows", None)
logging.getLogger('PIL').setLevel(logging.WARNING)
class OCTDataset(Dataset): 
    def __init__(self, args, data_type, infer_list=[]):
        self.file_list = {'train': glob.glob("{}/train/*".format(args.root_dirs)), 'test': glob.glob("{}/test/*".format(args.root_dirs))}
        self.mask_list = {'train': glob.glob("{}/train/*.png".format(args.mask_dir)), 'test': glob.glob("{}/test/*.png".format(args.mask_dir))}
        if data_type == 'inference':
            self.file_list = {'inference': ["{}/test/{}".format(args.root_dirs, item) for item in infer_list]}
            self.mask_list = {'inference': ["{}/test/{}.png".format(args.mask_dir, item.split('.')[0]) for item in infer_list]}
        self.labels_table = pd.read_csv("datasets/our_dataset/labels.csv")
        if args.combine_ez:
            self.labels_table['EZ'] = self.labels_table['EZ attenuated'] + self.labels_table['EZ disrupted']
            self.labels_table.loc[self.labels_table['EZ'] > 1, 'EZ'] = 1
        if 'BackGround' in OrgLabels:
            self.labels_table['BackGround'] = 1
            
        if data_type == 'train':
            self.transform = train_transform(args.is_size, False)
            self.transform_mask = train_transform(args.is_size, True)
        else:
            self.transform= valid_transform(args.is_size, False)
            self.transform_mask= valid_transform(args.is_size, True)
        self.roots = args.root_dirs
        # self.input_structure = args.input_structure
        self.data_type = data_type
        self.args = args
        self.normal_label = {'SRF': 0, 'IRF': 0, 'EZ attenuated': 0,  'EZ disrupted': 0,  'HRD': 0,  'RPE': 0,  'Retinal Traction': 0,  'Definite DRIL': 0,  'Questionable DRIL': 0,  'EZ': 0,  'BackGround': 1}

    def __getitem__(self, idx):
        data_path = sorted(self.file_list[self.data_type])[idx]
        mask_path = sorted(self.mask_list[self.data_type])[idx]
        image = Image.open(data_path)
        mask = np.asarray(Image.open(mask_path))
        # increase contrast of image
        if self.args.contrast:
            image = ImageEnhance.Contrast(image).enhance(2)
        image_arr = np.asarray(image)
        if (image_arr.ndim == 2):
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)
        image_name = data_path.split('/')[-1]
        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed)
        image_tensor = self.transform(image_arr)
        torch.manual_seed(seed)
        mask_tensor = self.transform_mask(mask)
        
        if image_name in self.labels_table['img'].values:
            labels_df = self.labels_table.loc[self.labels_table['img'] == image_name]
            labels = torch.FloatTensor([labels_df[x].to_numpy()[0] for x in OrgLabels])
        else:
            labels = torch.FloatTensor([self.normal_label[x] for x in OrgLabels])
        return {'image': image_tensor, 'labels': labels, 'path': data_path, 'mask': mask_tensor}
    def __len__(self):
        return len(self.file_list[self.data_type])
    
class RESCDataset(Dataset):
    def __init__(self, args, data_type, infer_list=[]):
        self.file_list = {'train': glob.glob("{}/train/original_images/*".format(args.root_dirs)), 'test': glob.glob("{}/valid/original_images/*".format(args.root_dirs))}
        self.mask_list = {'train': glob.glob("{}/train/*".format(args.mask_dir)), 'test': glob.glob("{}/valid/*".format(args.mask_dir))}
        self.labels_list = {'train': glob.glob("{}/train/label_images/*".format(args.root_dirs)), 'test': glob.glob("{}/valid/label_images/*".format(args.root_dirs))}

        if data_type == 'inference':
            self.file_list = {'inference': ["{}/valid/original_images/{}".format(args.root_dirs, item) for item in infer_list]}
            self.mask_list = {'inference': ["{}/valid/{}".format(args.mask_dir, item) for item in infer_list]}
            self.labels_list = {'inference': ["{}/valid/label_images/{}".format(args.root_dirs, item) for item in infer_list]}
        if data_type == 'train':
            self.transform = train_transform(args.is_size, False)
            self.transform_mask = train_transform(args.is_size, True)
        else:
            self.transform= valid_transform(args.is_size, False)
            self.transform_mask = valid_transform(args.is_size, True)
        self.roots = args.root_dirs
        self.data_type = data_type
        self.args = args

    def __getitem__(self, idx):
        data_path = sorted(self.file_list[self.data_type])[idx]
        mask_path = sorted(self.mask_list[self.data_type])[idx]
        label_path =  sorted(self.labels_list[self.data_type])[idx]
        
        image = Image.open(data_path)
        label_img = Image.open(label_path)
        mask = np.asarray(Image.open(mask_path))
        # increase contrast of image
        if self.args.contrast:
            image = ImageEnhance.Contrast(image).enhance(2)
        image_arr = np.asarray(image)
        label_arr = np.asarray(label_img)
        
        if (image_arr.ndim == 2):
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)
        if (label_arr.ndim == 2):
            label_arr = np.repeat(label_arr[..., np.newaxis], 3, -1)

        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed)
        image_tensor = self.transform(image_arr)
        torch.manual_seed(seed)
        mask_tensor = self.transform_mask(mask)
        torch.manual_seed(seed)
        label_tensor = self.transform_mask(label_arr)
        # vutils.save_image(label_tensor, 'waht3.png', normalize=False, scale_each=True)
        # import pdb; pdb.set_trace()
        
        # back: 0, ped: 128, srf: 191, retinal: 255
        l = {'SRF': 0, 'PED': 0, 'lesion': 0, 'health': 0, 'BackGround': 1}
        labels = np.unique(label_arr)
        if len(labels) == 1:
            l['health'] += 1
        if 128 in labels:
            l['PED'] += 1
        if 191 in labels:
            l['SRF'] += 1
        if 255 in labels:
            l['lesion'] +=1 

        return {'image': image_tensor, 'labels': torch.FloatTensor([l[x] for x in OrgLabels]), 'path': data_path, 'mask': mask_tensor,'annot': label_tensor}

    def __len__(self):
        return len(self.file_list[self.data_type])

class DukeDataset(Dataset):
    def __init__(self, args, data_type, infer_list=[]):
        train_from_resc_label_pth = 'baseline_models/SEAM/voc12/resc_cls_labels.npy'
        train_from_resc_label = np.load(train_from_resc_label_pth, allow_pickle=True)
        resc_srf_image_name_list = ''
        print(testt)
        ss
        self.file_list = {'train': glob.glob("{}/train/original_images/*".format(args.root_dirs)), 'test': glob.glob("{}/valid/original_images/*".format(args.root_dirs))}
        self.mask_list = {'train': glob.glob("{}/train/*".format(args.mask_dir)), 'test': glob.glob("{}/valid/*".format(args.mask_dir))}
        if data_type == 'inference':
            self.file_list = {'inference': ["{}/valid/original_images/{}".format(args.root_dirs, item) for item in infer_list]}
            self.mask_list = {'inference': ["{}/valid/{}".format(args.mask_dir, item) for item in infer_list]}
        self.labels_list = {'train': glob.glob("{}/train/label_images/*".format(args.root_dirs)), 'test': glob.glob("{}/valid/label_images/*".format(args.root_dirs))}
        
        if data_type == 'train':
            self.transform = train_transform(args.is_size, False)
            self.transform_mask = train_transform(args.is_size, True)
        else:
            self.transform= valid_transform(args.is_size, False)
            self.transform_mask = valid_transform(args.is_size, True)
        self.roots = args.root_dirs
        self.data_type = data_type
        self.args = args

    def __getitem__(self, idx):
        data_path = sorted(self.file_list[self.data_type])[idx]
        mask_path = sorted(self.mask_list[self.data_type])[idx]
        label_path =  sorted(self.labels_list[self.data_type])[idx]
        
        image = Image.open(data_path)
        label_img = Image.open(label_path)
        mask = np.asarray(Image.open(mask_path))
        # increase contrast of image
        if self.args.contrast:
            image = ImageEnhance.Contrast(image).enhance(2)
        image_arr = np.asarray(image)
        label_arr = np.asarray(label_img)
        
        if (image_arr.ndim == 2):
            image_arr = np.repeat(image_arr[..., np.newaxis], 3, -1)
        if (label_arr.ndim == 2):
            label_arr = np.repeat(label_arr[..., np.newaxis], 3, -1)

        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed)
        image_tensor = self.transform(image_arr)
        torch.manual_seed(seed)
        mask_tensor = self.transform_mask(mask)
        torch.manual_seed(seed)
        label_tensor = self.transform_mask(label_arr)
        import pdb; pdb.set_trace()
        # vutils.save_image(label_tensor, 'waht3.png', normalize=False, scale_each=True)
        
        # back: 0, ped: 128, srf: 191, retinal: 255
        l = {'SRF': 0, 'PED': 0, 'lesion': 0, 'health': 0, 'BackGround': 1}
        labels = np.unique(label_arr)
        if len(labels) == 1:
            l['health'] += 1
        if 128 in labels:
            l['PED'] += 1
        if 191 in labels:
            l['SRF'] += 1
        if 255 in labels:
            l['lesion'] +=1 

        return {'image': image_tensor, 'labels': torch.FloatTensor([l[x] for x in OrgLabels]), 'path': data_path, 'mask': mask_tensor,'annot': label_tensor}
    def __len__(self):
        return len(self.file_list[self.data_type])
 
def train_transform(is_size, is_mask):
    inter = Image.NEAREST if is_mask else Image.BILINEAR
    transform_seq = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(is_size, interpolation=inter),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=60),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #  std=[0.229, 0.224, 0.225])
    ])
    return transform_seq

def valid_transform(is_size, is_mask):
    inter = Image.NEAREST if is_mask else Image.BILINEAR
    transform_seq = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(is_size, interpolation=inter),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
    return transform_seq

if __name__ == "__main__":
    # root_dirs = ["datasets/our_dataset/original/DME_1", "datasets/our_dataset/original/DME_2", "datasets/our_dataset/original/DME_3", "datasets/our_dataset/original/DME_4", "datasets/our_dataset/original/DR"]
    # labels_table = []
    # for root_dir in root_dirs:
    #     labels_table.append(pd.read_csv("%s/labels.csv" % root_dir)) 
    # labels_table = pd.concat(labels_table, ignore_index=True)
    # labels_table.to_csv('datasets/our_dataset/labels.csv', index=False)
    # print(labels_table)

    # dataset = OCTDataset(root_dirs, transform=valid_transform())
    # acc, f1m, f1mi = 0, 0, 0
    # gt = [0.0, 1.0, ]#0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # gt_dr = [0.0, 1.0,]# 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    # for data in dataset:
    #     if (data["path"].split('/')[0] == "dataset_DR" ): 
    #         gt_f = gt_dr
    #     else: gt_f = gt
    #     acc +=  accuracy_score(gt_f,data["labels"].numpy())
    #     f1m += f1_score(gt_f,data["labels"].numpy(),average = 'macro', zero_division=1)
    #     f1mi += f1_score(gt_f,data["labels"].numpy(),average = 'micro', zero_division=1)
    # print(acc / len(dataset), f1m / len(dataset), f1mi / len(dataset))
    
    '''
    Train: {'srf': 2934, 'ped': 402, 'retinal': 4664, 'health': 4296}
    Test: {'srf': 360, 'ped': 41, 'retinal': 834, 'health': 1086}
    '''
    class Args():
        def __init__(self) -> None:
            self.root_dirs = 'RESC'
            self.mask_dir = 'RESC/mask'
            self.contrast = False
            self.is_size = (256, 256)
    args = Args()
    # resc_dataset = RESCDataset(args, 'test')
    # l = {'srf': 0, 'ped': 0, 'retinal': 0, 'health': 0}
    # for idx in range(0, len(resc_dataset)):
    #     res = resc_dataset[idx]
    #     print(res)
    #     import pdb; pdb.set_trace()
    
    duke_dataset = DukeDataset(args, 'test')
    


# %%
