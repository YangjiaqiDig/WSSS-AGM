""" Options

This script is largely based on junyanz/pytorch-CycleGAN-and-pix2pix.

Returns:
    [argparse]: Class containing argparse
"""

import argparse
import torch

class Configs():
    """Configs class

    Returns:
        [argparse]: argparse containing train and test Configs
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--root_dirs", type=str, default=["our_dataset/dataset_DR",
                                                            "our_dataset/dataset_DME/1", 
                                                            "our_dataset/dataset_DME/2",
                                                            "our_dataset/dataset_DME/3",
                                                            "our_dataset/dataset_DME/4"],
                            help="list of directories")
        self.parser.add_argument("--k_folds", type=int,
                            default=10, help="k folds")
        self.parser.add_argument("--save_folder", type=str, default="outputs/results_back_class",
                            help="Path or url of the dataset")
        self.parser.add_argument("--train_batch_size", type=int,
                            default=8, help="Batch size for training")
        self.parser.add_argument("--valid_batch_size", type=int,
                            default=1, help="Batch size for validation")
        self.parser.add_argument("--lr", type=float,
                            default=0.0001, help="Learning rate")
        self.parser.add_argument("--cam_type", type=str, default="gradcam", help="GradCAM")
        self.parser.add_argument("--n_epochs", type=int, default=25, #25
                            help="Number of training epochs")
        self.parser.add_argument("--refine_epoch_point", type=int, default=5, 
                            help="The start epoch for refine input by cam")
        self.parser.add_argument("--check_point", type=str, default="/fold-0/25.pwf",
                            help="Path of the pre-trained Network")
        self.parser.add_argument("--continue_train", type=bool, default=True, help="Continue train")
        self.parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
        self.parser.add_argument("--backbone", type=str, default="resnet18", help="resnet18, resnet50, resnet101, vgg16")
        self.parser.add_argument("--input_gan", type=bool, default=True, help="If involve GANs generation as input")
        self.parser.add_argument("--input_structure", type=bool, default=False, help="If involve Struture as input")
        self.parser.add_argument("--combine_ez", type=bool, default=True, help="If combine ez two conditions")
        self.parser.add_argument("--is_size", default=(256, 256), help="resize of input image, need same size as GANs generation") #(500,750)
        self.parser.add_argument("--inference", type=bool, default=False, help="Inference only")
        self.parser.add_argument("--save_inference", type=str, default='outputs_inference', help="Save inference or test images directory")
        self.parser.add_argument("--num_iteration", type=int, default=10, help="Number of iterations for refine CAM")

    def parse(self):
        args = self.parser.parse_args()
        return args
        
