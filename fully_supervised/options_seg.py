""" Options
Returns:
    [argparse]: Class containing argparse
"""

import argparse
import torch
import os


class Configs:
    """Configs class

    Returns:
        [argparse]: argparse containing train and test Configs
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        """Dataset params"""
        self.parser.add_argument(
            "--root_dirs",
            type=str,
            default="datasets/RESC",
            help="root datasets directory: 2015_BOE_Chiu | RESC | our_dataset",
        )
        self.parser.add_argument(
            "--save_name",
            type=str,
            default="seg_unet_layer_v2",
            help="Path or url of the dataset",
        )
        self.parser.add_argument(
            "--labels",
            type=str,
            default=["BackGround", "SRF", "IRF", "EZ disrupted", "HRD"],
            help="['SRF', 'IRF', 'EZ', 'HRD',  'RPE', 'BackGround', 'EZ attenuated', 'EZ disrupted', 'Retinal Traction', 'Definite DRIL']",
        )
        self.parser.add_argument(
            "--resc_labels",
            type=str,
            default=["BackGround", "SRF", "PED"],
            help="['SRF', 'PED', 'LESION', 'BackGround']",
        )
        self.parser.add_argument(
            "--boe_labels",
            type=str,
            default=["BackGround", "Fluid"],
            help="['Fluid', 'BackGround']",
        )
        self.parser.add_argument(
            "--save_results",
            type=bool,
            default=True,
            help="Save the results in save folder or not",
        )
        self.parser.add_argument(
            "--expert_annot", type=str, default="both", help="mina, meera, both"
        )

        """Model achitecture params"""
        self.parser.add_argument(
            "--retinal_model",
            type=str,
            default="UNet",
            help="UNet | Segformer",
        )

        """Trainning params"""
        self.parser.add_argument(
            "--loss_type", type=str, default="layer_ped", help="ce | layer | layer_ped"
        )
        self.parser.add_argument(
            "--layer_scale", type=float, default=1, help="Layer loss scale factor"
        )
        self.parser.add_argument(
            "--epochs", type=int, default=150, help="Number of training epochs"
        )
        self.parser.add_argument(
            "--train_batch_size", type=int, default=4, help="Batch size for training"
        )
        self.parser.add_argument(
            "--valid_batch_size", type=int, default=1, help="Batch size for validation"
        )
        self.parser.add_argument(
            "--lr", type=float, default=0.001, help="Classification Learning rate"
        )  # 0.001
        self.parser.add_argument(
            "--lr_schedule",
            type=dict,
            default={"step": 50, "gamma": 0.1},
            help="Learning rate decay step and gamma",
        )
        self.parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device (cuda or cpu)",
        )

        """Inference or continue fine-tuning params"""
        self.parser.add_argument(
            "--continue_train", type=bool, default=False, help="Continue train"
        )
        self.parser.add_argument(
            "--check_point",
            type=str,
            default="outputs/resc/seg_unet",
            help="Path of the pre-trained Network",
        )
        self.parser.add_argument(
            "--ckp_epoch", type=str, default="50", help="best, last, best_iou"
        )
        self.parser.add_argument(
            "--is_size",
            default=(640, 640),
            help="resize of input image, need same size as GANs generation",
        )
        self.parser.add_argument(
            "--save_inference",
            type=str,
            default="outputs_inference",
            help="Save inference or test images directory",
        )

    def parse(self, is_inference=False):
        args = self.parser.parse_args()
        if "RESC" in args.root_dirs:
            args.annot_dir = "valid/label_images"
            args.mask_dir = "datasets/RESC/mask"
            args.save_folder = "outputs/resc/" + args.save_name
        elif "BOE" in args.root_dirs:
            args.annot_dir = "segment_annotation/labels"
            args.save_folder = "outputs/duke/" + args.save_name
        else:
            if args.expert_annot == "both":
                args.annot_dir = "annot_combine"
            else:
                args.annot_dir = "annotation_v2"
            args.mask_dir = "datasets/our_dataset/mask"
            args.save_folder = (
                "outputs/our_dataset/" + args.expert_annot + args.save_name
            )

        if is_inference:
            print("Start inference args parsing")
            return args

        save_opt_name = "opt.txt"
        if args.continue_train:
            args.save_folder = args.check_point
            save_opt_name = "opt_continue.txt"

        file_name = os.path.join(args.save_folder, save_opt_name)
        if not os.path.exists(args.save_folder):
            os.makedirs(args.save_folder)
        with open(file_name, "wt") as opt_file:
            opt_file.write("------------ Options -------------\n")
            for k, v in sorted(vars(args).items()):
                opt_file.write("%s: %s\n" % (str(k), str(v)))
            opt_file.write("-------------- End ----------------\n")

        return args

    def get_labels(self):
        if "RESC" in self.parser.parse_args().root_dirs:
            return self.parser.parse_args().resc_labels
        if "BOE" in self.parser.parse_args().root_dirs:
            return self.parser.parse_args().boe_labels
        return self.parser.parse_args().labels


if __name__ == "__main__":
    args = Configs().parse(False)
    print(args)
