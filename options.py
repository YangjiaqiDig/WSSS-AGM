""" Options
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
        '''Dataset params'''
        self.parser.add_argument("--root_dirs", type=str, default="datasets/our_dataset", help="root datasets directory: 2015_BOE_Chiu | RESC | our_dataset")
        self.parser.add_argument("--save_name", type=str, default="att_orig_gan", help="Path or url of the dataset")
        self.parser.add_argument("--labels", type=str, default=['SRF', 'IRF', 'EZ disrupted', 'HRD', 'BackGround'], help="['SRF', 'IRF', 'EZ', 'HRD',  'RPE', 'BackGround', 'EZ attenuated', 'EZ disrupted', 'Retinal Traction', 'Definite DRIL']")
        self.parser.add_argument("--resc_labels", type=str, default=['SRF', 'PED', 'BackGround'], help="['SRF', 'PED', 'LESION', 'BackGround']")        
        self.parser.add_argument("--boe_labels", type=str, default=['Fluid', 'BackGround'], help="['Fluid', 'BackGround']")
        self.parser.add_argument("--save_cam_images", type=bool, default=False, help="Save the results in save folder or not")
        self.parser.add_argument("--expert_annot", type=str, default='both', help="mina, meera, both")
        
        '''Model achitecture params'''
        self.parser.add_argument("--input_gan", type=bool, default=True, help="If involve GANs generation as input")
        self.parser.add_argument("--att_module", type=bool, default=True, help="If activate self-attention branch")        
        self.parser.add_argument("--backbone", type=str, default="resnet50", help="resnet18, resnet50, resnet101, vgg16")
        self.parser.add_argument("--retinal_model", type=str, default="MultiTaskModel_att", help="MultiTaskModel_att | CNNs | Transformers")
        self.parser.add_argument("--model_gan", type=str, default='pretrained/gan/regular_512/best', help="Pretrained GAN Model path")
        self.parser.add_argument("--cam_type", type=str, default="gradcam", help="GradCAM")
        self.parser.add_argument("--aug_smooth", type=bool, default=False, help="If make grad-cam aug smooth")
        
        '''Trainning params'''
        self.parser.add_argument("--w_ce", type=float, default=1, help="Cross Entropy loss weight")
        self.parser.add_argument("--w_dice", type=float, default=0, help="Dice loss weight")
        self.parser.add_argument("--reg_epochs", type=int, default=50, help="Number of training epochs")
        self.parser.add_argument("--seg_epochs", type=int, default=0, help="Number of segmentation epochs")
        self.parser.add_argument("--iter_epochs", type=int, default=0, help="Number of iterations for refine CAM")
        self.parser.add_argument("--train_batch_size", type=int,default=4, help="Batch size for training")
        self.parser.add_argument("--valid_batch_size", type=int,default=1, help="Batch size for validation")
        self.parser.add_argument("--lr", type=float, default=0.001, help="Classification Learning rate") # 0.001
        self.parser.add_argument("--lr_schedule", type=dict, default={'step': 200, 'gamma': 0.5}, help="Learning rate decay step and gamma")
        self.parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
        
        '''Inference or continue fine-tuning params'''
        self.parser.add_argument("--check_point", type=str, default="outputs/our_dataset/att_orig_gan", help="Path of the pre-trained Network")
        self.parser.add_argument("--ckp_epoch", type=str, default="iter_29", help="best, last, best_iou")        
        self.parser.add_argument("--continue_train", type=bool, default=False, help="Continue train")
        self.parser.add_argument("--out_cam_pred_alpha", type=int, default=0.8, help="bg score") # # 0.7 resc 0.8 duke 0.8 ours
        self.parser.add_argument("--is_size", default=(512, 512), help="resize of input image, need same size as GANs generation")
        self.parser.add_argument("--save_inference", type=str, default='outputs_inference', help="Save inference or test images directory")
        
    def parse(self, is_inference=False):
        args = self.parser.parse_args()
        if 'RESC' in args.root_dirs:
            args.annot_dir = 'valid/label_images'
            args.mask_dir = "datasets/RESC/mask"
            args.save_folder = 'outputs/resc_512/' + args.save_name
        elif 'BOE' in args.root_dirs:
            args.annot_dir = "segment_annotation/labels"
            args.save_folder = 'outputs/duke/' + args.save_name
        else:
            if args.expert_annot == 'both':
                args.annot_dir = 'annot_combine'
            else: args.annot_dir = 'annotation_v2'
            args.mask_dir = "datasets/our_dataset/mask"
            args.save_folder = 'outputs/our_dataset/' + args.expert_annot + args.save_name
            
        if is_inference:
            print('Start inference args parsing')
            return args
        
        save_opt_name = 'opt.txt'
        if args.continue_train:
            args.save_folder = args.check_point
            save_opt_name = 'opt_continue.txt'
            
        file_name = os.path.join(args.save_folder, save_opt_name)
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
            return self.parser.parse_args().boe_labels
        return self.parser.parse_args().labels
        
if __name__ == "__main__":
    args = Configs().parse(False)
    print(args)