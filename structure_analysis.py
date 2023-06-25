import argparse
import cv2
import torchvision.utils as vutils
from utils.utils import post_process_cam

import torch
from PIL import Image

from tqdm import tqdm
from torchvision import transforms

from dataset import (DukeDataset, OCTDataset, RESCDataset)
import time
import os
from preprocess import generate_background_mask_for_GAN
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE_NR = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NR

from gan_inference import load_gan_model
class Inference():
    def __init__(self, args):
        with torch.no_grad():
            path = "pretrained/gan/regular_512/best/netG.pth"
            self.gan_pretrained = load_gan_model(path, DEVICE_NR)
            print(f' Loaded Pretained GAN weights from {path}.')
        self.transform_norml = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.args = args

    def get_dataset(self, data_type='resc'):
        if data_type == 'oct':
            dataset_train = OCTDataset(self.args, data_type='train')
            dataset_test = OCTDataset(self.args, data_type='test')
        elif data_type == 'resc':
            dataset_train = RESCDataset(self.args, data_type='train')
            dataset_test = RESCDataset(self.args, data_type='test')
        else:
            dataset_train = DukeDataset(self.args, data_type='train')
            dataset_test = DukeDataset(self.args, data_type='test')
        
        return dataset_train, dataset_test
    
    def get_masks_and_save(self, img_name, data, resized_back_img):
        if not os.path.exists(f'structure_analysis/{img_name}'):
            os.mkdir(f'structure_analysis/{img_name}')
        vutils.save_image(
            resized_back_img.squeeze(0),
            f'structure_analysis/{img_name}/healthy.png',
            normalize=False,
            scale_each=False,
        )
        healthy_mask = generate_background_mask_for_GAN(resized_back_img.squeeze(0))
        cv2.imwrite(f'structure_analysis/{img_name}/healthy_mask.png', healthy_mask)

        # cv2 overlay 2 images
        origin_image = Image.open(f"{data['path'][0]}").convert('RGB')
        origin_image = np.copy(np.asarray(origin_image))
        origin_mask = np.asarray(Image.open(f"{data['mask_path'][0]}").convert('RGB'))
        color_mask = np.zeros_like(origin_mask)
        color_mask[np.where((origin_mask==[255, 255,255]).all(axis=2))] = [0,255,0]
        overlayed_mask = cv2.addWeighted(origin_image.astype(np.uint8), 1, color_mask, 0.1, 0)
        cv2.imwrite(f'structure_analysis/{img_name}/original_overlay.png', overlayed_mask)

        color_mask = np.zeros_like(healthy_mask)
        color_mask[np.where((healthy_mask==[255, 255,255]).all(axis=2))] = [0,255,0]
        overlayed_healthy_mask = cv2.addWeighted((resized_back_img[0].cpu().numpy().transpose(1, 2, 0)* 255).astype(np.uint8), 1, color_mask, 0.1, 0)
        cv2.imwrite(f'structure_analysis/{img_name}/healthy_overlay.png', overlayed_healthy_mask)

        cross_overlay = cv2.addWeighted(origin_image, 1, color_mask, 0.1, 0)
        cv2.imwrite(f'structure_analysis/{img_name}/cross_overlay.png', cross_overlay)

    def inference(self, infer_list=[], data_type='resc'):
        list_healthy_img = []
        if not infer_list:
            _, infer_dataset = self.get_dataset(data_type)
        else:
            if data_type == 'oct':
                infer_dataset = OCTDataset(self.args, data_type='inference', infer_list=infer_list)
            elif data_type == 'resc':
                infer_dataset = RESCDataset(self.args, data_type='inference', infer_list=infer_list)
            elif data_type == 'duke':
                infer_dataset = DukeDataset(self.args, data_type='inference', infer_list=infer_list)
        dataloader = torch.utils.data.DataLoader(
            infer_dataset,
            num_workers=8,
            batch_size=1, shuffle=False)
        
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image, labels, mask, shape = data["image"].to(self.args.device), data["labels"].to(self.args.device), data['mask'].to(self.args.device), data['shape']    
            img_name = data["path"][0].split('/')[-1].split('.')[0]
            updated_image = image.clone()
            with torch.no_grad():
                gan_inputs = self.transform_norml(updated_image)
                healthy_img = self.gan_pretrained.inference(gan_inputs)
                list_healthy_img.append(healthy_img)
            normalized_healthy_image = post_process_cam(healthy_img[0].cpu().numpy())
            resized_back_healthy_version = torch.nn.functional.interpolate(torch.from_numpy(normalized_healthy_image).unsqueeze(0), size=(shape[0], shape[1]), mode='bilinear', align_corners=True)
            # self.get_masks_and_save(img_name, data, resized_back_healthy_version)
            edges = cv2.Canny(resized_back_healthy_version,100,200)

            
            
        return list_healthy_img
    

if __name__ == "__main__":
    is_inference=True
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dirs", type=str, default="datasets/RESC", help="root datasets directory: 2015_BOE_Chiu | RESC | our_dataset")
    parser.add_argument("--labels", type=str, default=['SRF', 'IRF', 'EZ disrupted', 'HRD', 'BackGround'], help="['SRF', 'IRF', 'EZ', 'HRD',  'RPE', 'BackGround', 'EZ attenuated', 'EZ disrupted', 'Retinal Traction', 'Definite DRIL']")
    parser.add_argument("--resc_labels", type=str, default=['SRF', 'PED', 'BackGround'], help="['SRF', 'PED', 'LESION', 'BackGround']")        
    parser.add_argument("--boe_labels", type=str, default=['Fluid', 'BackGround'], help="['Fluid', 'BackGround']")
    parser.add_argument("--expert_annot", type=str, default='both', help="mina, meera, both")
    parser.add_argument("--is_size", default=(512, 512), help="resize of input image, need same size as GANs generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    args = parser.parse_args()
    if 'RESC' in args.root_dirs:
        args.mask_dir = "datasets/RESC/mask"
    elif 'BOE' in args.root_dirs:
        args.annot_dir = "segment_annotation/labels"
        args.save_folder = 'outputs/duke/' + args.save_name
    else:
        # if args.expert_annot == 'both':
        #     args.annot_dir = 'annot_combine'
        # else: args.annot_dir = 'annotation_v2'
        args.mask_dir = "datasets/our_dataset/mask"
    validator = Inference(args)
    start = time.time()
    # num_examples = validator.prepare_pesudo_label_for_seg()
    validator.inference(infer_list=['sn22698_124.bmp'], data_type='resc')
    # validator.inference(infer_list=['DR10.jpeg', 'DR91.jpeg', 'NORMAL-76914-1.jpeg', 'DME-3565572-7.jpeg', 'NORMAL-2709055-1.jpeg', 'DME-4240465-13.jpeg'], data_type='oct')


    # validator.inference(infer_list=['DME-15307-1.jpeg',
    #                               'DME-4240465-41.jpeg', 
    #                               'DR10.jpeg',
    #                               'NORMAL-15307-1.jpeg'])

# %%
