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
            default="../datasets/2015_BOE_Chiu",
            help="root datasets directory: 2015_BOE_Chiu | RESC | our_dataset",
        )
        self.parser.add_argument(
            "--save_name", type=str, default="caption_abnormal_v5", help="Path or url of the dataset"
        )
        self.parser.add_argument(
            "--nyu_labels",
            type=str,
            default=["BackGround", "SRF", "IRF", "EZ disrupted", "HRD"],
            help="['SRF', 'IRF', 'EZ', 'HRD',  'RPE', 'BackGround', 'EZ attenuated', 'EZ disrupted', 'Retinal Traction', 'Definite DRIL']",
        )
        self.parser.add_argument(
            "--resc_labels",
            type=str,
            default=["BackGround", "SRF", "PED"],  # "BackGround",
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
            default=False,
            help="Save the results in save folder or not",
        )
        self.parser.add_argument(
            "--expert_annot", type=str, default="both", help="mina, meera, both"
        )

        """Trainning params"""
        self.parser.add_argument(
            "--freeze_layers",
            type=str,
            default="block3",
            help="before layer need to freeze",
        )
        self.parser.add_argument(
            "--clip_branch", type=bool, default=True, help="include layer branch"
        )
        self.parser.add_argument(
            "--clip_version", type=str, default="large", help="base or large"
        )
        self.parser.add_argument(
            "--caption_branch", type=bool, default=True, help="include caption branch"
        )
        self.parser.add_argument(
            "--caption_version",
            type=str,
            default="blip_norm_clip_base_embed",
            help="blip_norm_clip_base_embed, blip_norm_clip_large_embed, blip_minilm_embed, vit_clip_base_embed, vit_clip_large_embed or vit_minilm_embed",
        )
        self.parser.add_argument(
            "--add_abnormal",
            type=bool,
            default=True,
            help="add anomalous image on layer branch",
        )
        self.parser.add_argument(
            "--layer_branch", type=bool, default=True, help="include layer branch"
        )
        self.parser.add_argument(
            "--n_layer_channels", type=int, default=3, help="12 or 3"
        )
        self.parser.add_argument(
            "--constraint_loss",
            type=bool,
            default=False,
            help="segmentation branch generates classification loss",
        )
        self.parser.add_argument(
            "--pool_type", type=str, default="max", help="max or avg"
        )
        self.parser.add_argument(
            "--n_epochs", type=int, default=30, help="Number of training epochs"
        )
        self.parser.add_argument(
            "--seg_start_epoch", type=int, default=10000, help="Number of classes"
        )
        self.parser.add_argument(
            "--train_batch_size", type=int, default=8, help="Batch size for training"
        )
        self.parser.add_argument(
            "--valid_batch_size", type=int, default=8, help="Batch size for validation"
        )
        self.parser.add_argument(
            "--optimizer", type=str, default="adam", help="adam or adamw"
        )
        self.parser.add_argument(
            "--lr", type=float, default=1e-4, help="Classification Learning rate"
        )  # 0.001 # 0.0001 ref # 0.000015
        self.parser.add_argument(
            "--lr_schedule",
            type=dict,
            default={"step": 10000000, "gamma": 0.1},
            help="Learning rate decay step and gamma",
        )
        self.parser.add_argument(
            "--warmup_ratio", type=float, default=0.05, help="warmup iterations rate"
        )
        self.parser.add_argument(
            "--device",
            type=str,
            default="cuda" if torch.cuda.is_available() else "cpu",
            help="Device (cuda or cpu)",
        )

        """Inference or continue fine-tuning params"""
        self.parser.add_argument(
            "--check_point",
            type=str,
            default="outputs/resc/caption_abnormal_v5",
            help="Path of the pre-trained Network",
        )
        self.parser.add_argument(
            "--ckp_epoch", type=str, default="best_iou", help="best, last, best_iou"
        )  # ours (v2) iter_39 , duke iter_27 or iter_21?, resc 17
        self.parser.add_argument(
            "--load_model", type=bool, default=False, help="Loading pretraiined train"
        )
        self.parser.add_argument(
            "--out_cam_pred_alpha", type=int, default=0.7, help="bg score"
        )  # # 0.7 resc 0.8 duke 0.7 ours
        self.parser.add_argument(
            "--is_size",
            default=(512, 512),
            help="resize of input image, need same size as layer generation - 224, 224",
        )
        self.parser.add_argument(
            "--save_inference",
            type=str,
            default="outputs_test",
            help="Save inference or test images directory",
        )

    def parse(self, is_inference=False):
        args = self.parser.parse_args()
        if "RESC" in args.root_dirs:
            args.annot_dir = "valid/label_images"
            args.mask_dir = "../datasets/RESC/mask"
            args.save_folder = "outputs/resc/" + args.save_name
        elif "BOE" in args.root_dirs:
            args.annot_dir = "segment_annotation/labels"
            args.save_folder = "outputs/duke/" + args.save_name
        else:
            if args.expert_annot == "both":
                args.annot_dir = "annot_combine"
            else:
                args.annot_dir = "annotation_v2"
            args.mask_dir = "../datasets/our_dataset/mask"
            args.save_folder = (
                "outputs/our_dataset/" + args.expert_annot + "_" + args.save_name
            )

        if is_inference:
            print("Start inference args parsing")
            return args

        save_opt_name = "opt.txt"
        if args.load_model:
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
        return self.parser.parse_args().nyu_labels


if __name__ == "__main__":
    args = Configs().parse(False)
    print(args)
