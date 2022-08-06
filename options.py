""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import torch
import os

class Configs():
    """Configs class

    Returns:
        [argparse]: argparse containing train and test Configs
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--root_dirs", type=str, default="datasets/RESC", help="root datasets directory")
        self.parser.add_argument("--mask_dir", type=str, default="datasets/RESC/mask", help="Retinal mask datasets directory")
        self.parser.add_argument("--save_folder", type=str, default="outputs/resc_512/origin_gan", help="Path or url of the dataset")
        self.parser.add_argument("--labels", type=str, default=['SRF', 'IRF', 'EZ disrupted', 'HRD', 'BackGround'], help="['SRF', 'IRF', 'EZ', 'HRD',  'RPE', 'BackGround', 'EZ attenuated', 'EZ disrupted', 'Retinal Traction', 'Definite DRIL']")
        self.parser.add_argument("--resc_labels", type=str, default=['SRF', 'PED', 'BackGround'], help="['SRF', 'PED', 'LESION', 'BackGround']")        
        self.parser.add_argument("--boe_labels", type=str, default=['Fluid', 'BackGround'], help="['SRF', 'PED', 'LESION', 'BackGround']")
        self.parser.add_argument("--contrast", type=bool, default=False, help="Increase contrast of input image")
        self.parser.add_argument("--mask_enhance", type=bool, default=False, help="Mask input image by enhance")
        
        self.parser.add_argument("--segmentation", type=int, default=50, help="The number of epoch that starts segmentation branch")
        self.parser.add_argument("--input_gan", type=bool, default=True, help="If involve GANs generation as input")
        self.parser.add_argument("--w_ce", type=float, default=1, help="Cross Entropy loss weight")
        self.parser.add_argument("--w_dice", type=float, default=0, help="Dice loss weight")
        self.parser.add_argument("--n_epochs", type=int, default=40, help="Number of training epochs")
        self.parser.add_argument("--num_iteration", type=int, default=0, help="Number of iterations for refine CAM")
        
        self.parser.add_argument("--train_batch_size", type=int,default=8, help="Batch size for training")
        self.parser.add_argument("--valid_batch_size", type=int,default=1, help="Batch size for validation")
        self.parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
        self.parser.add_argument("--lr_schedule", type=dict, default={'step': 15, 'gamma': 0.5}, help="Learning rate decay step and gamma")
        
        self.parser.add_argument("--check_point", type=str, default="outputs/resc_512_v2/origin_gan", help="Path of the pre-trained Network")
        self.parser.add_argument("--continue_train", type=bool, default=False, help="Continue train")
        self.parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
        self.parser.add_argument("--cam_type", type=str, default="gradcam", help="GradCAM")
        self.parser.add_argument("--backbone", type=str, default="resnet50", help="resnet18, resnet50, resnet101, vgg16")
        self.parser.add_argument("--model_gan", type=str, default='pretrained/gan/regular_512/best', help="Pretrained GAN Model path")
        
        self.parser.add_argument("--combine_ez", type=bool, default=False, help="If combine ez two conditions")
        self.parser.add_argument("--is_size", default=(512, 512), help="resize of input image, need same size as GANs generation") #(500,750)
        self.parser.add_argument("--save_inference", type=str, default='outputs_inference', help="Save inference or test images directory")
        
        ''' currently deprecated '''
        # self.parser.add_argument("--k_folds", type=int, default=10, help="k folds")
        # self.parser.add_argument("--input_structure", type=bool, default=False, help="If involve Struture as input")
        self.parser.add_argument("--n_refine_background", type=int, default=0, help="Number of background only refinement")
    def parse(self, is_inference=False):
        args = self.parser.parse_args()
        if is_inference:
            print('Start inference args parsing')
            return args
        file_name = os.path.join(args.save_folder, 'opt.txt')
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(vars(args).items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return args
   
    def get_labels(self):
        if 'RESC' in self.parser.parse_args().root_dirs:
            return self.parser.parse_args().resc_labels
        if 'BOE' in self.parser.parse_args().root_dirs:
            return self.parser.parse_args.boe_labels
        return self.parser.parse_args().labels
        
