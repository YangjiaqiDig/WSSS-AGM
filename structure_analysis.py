import argparse
import cv2
import torchvision.utils as vutils

import torch
from PIL import Image

from tqdm import tqdm
from torchvision import transforms

from anomaly_guided.dataset import DukeDataset, OCTDataset, RESCDataset
import time
import os
from anomaly_guided.preprocess import generate_background_mask_for_GAN
import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE_NR = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NR

from anomaly_guided.gan_inference import load_gan_model

def get_gan_path_by_image_path(image_path):
    if "BOE" in image_path:
        mask_path = image_path.replace("/images/", "/gan_healthy/")
    elif "RESC" in image_path:
        mask_path = image_path.replace("/original_images/", "/gan_healthy/")
    elif "NORMAL" in image_path:
        mask_path = image_path.replace("train/0.normal", "train/gan_healthy")
    else:
        mask_path = image_path.replace("original", "gan_healthy")
    return mask_path

def normalized_batch_tensor(t):
    orig_size = t.shape
    t = t.view(orig_size[0], -1)
    t -= t.min(1, keepdim=True)[0]
    t /= t.max(1, keepdim=True)[0]
    t = t.view(orig_size)
    return t


def diff_map_for_att(orig_tensor, gan_tensor, mask_tensor=None):
    # batch, channel, h, w
    normalized_orig = orig_tensor.clone()
    normalized_gan = gan_tensor.clone()

    normalized_orig = normalized_batch_tensor(normalized_orig)
    normalized_gan = normalized_batch_tensor(normalized_gan)

    abs_diff = torch.abs(normalized_orig - normalized_gan)
    if mask_tensor is None:
        return abs_diff

    mask_out_diff = abs_diff * mask_tensor

    return mask_out_diff

class Inference:
    def __init__(self, args):
        with torch.no_grad():
            path = "anomaly_guided/pretrained/gan/regular_512/best/netG.pth"
            self.gan_pretrained = load_gan_model(path, DEVICE_NR)
            print(f" Loaded Pretained GAN weights from {path}.")
        self.transform_norml = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.args = args

    def get_dataset(self, data_type="resc"):
        if data_type == "oct":
            dataset_train = OCTDataset(self.args, data_type="train", is_generate_pseudo_label=True)
            dataset_test = OCTDataset(self.args, data_type="test")
        elif data_type == "resc":
            dataset_train = RESCDataset(self.args, data_type="train", is_generate_pseudo_label=True)
            dataset_test = RESCDataset(self.args, data_type="test")
        else:
            dataset_train = DukeDataset(self.args, data_type="train", is_generate_pseudo_label=True)
            dataset_test = DukeDataset(self.args, data_type="test")

        return dataset_train, dataset_test

    def get_masks_and_save(self, img_name, data, resized_back_img):
        if not os.path.exists(f"structure_analysis/{img_name}"):
            os.mkdir(f"structure_analysis/{img_name}")
        vutils.save_image(
            resized_back_img.squeeze(0),
            f"structure_analysis/{img_name}/healthy.png",
            normalize=False,
            scale_each=False,
        )
        healthy_mask = generate_background_mask_for_GAN(resized_back_img.squeeze(0))
        cv2.imwrite(f"structure_analysis/{img_name}/healthy_mask.png", healthy_mask)

        # cv2 overlay 2 images
        origin_image = Image.open(f"{data['path'][0]}").convert("RGB")
        origin_image = np.copy(np.asarray(origin_image))
        origin_mask = np.asarray(Image.open(f"{data['mask_path'][0]}").convert("RGB"))
        color_mask = np.zeros_like(origin_mask)
        color_mask[np.where((origin_mask == [255, 255, 255]).all(axis=2))] = [0, 255, 0]
        overlayed_mask = cv2.addWeighted(
            origin_image.astype(np.uint8), 1, color_mask, 0.1, 0
        )
        cv2.imwrite(
            f"structure_analysis/{img_name}/original_overlay.png", overlayed_mask
        )

        color_mask = np.zeros_like(healthy_mask)
        color_mask[np.where((healthy_mask == [255, 255, 255]).all(axis=2))] = [
            0,
            255,
            0,
        ]
        overlayed_healthy_mask = cv2.addWeighted(
            (resized_back_img[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(
                np.uint8
            ),
            1,
            color_mask,
            0.1,
            0,
        )
        cv2.imwrite(
            f"structure_analysis/{img_name}/healthy_overlay.png", overlayed_healthy_mask
        )

        cross_overlay = cv2.addWeighted(origin_image, 1, color_mask, 0.1, 0)
        cv2.imwrite(f"structure_analysis/{img_name}/cross_overlay.png", cross_overlay)

    def inference(self, infer_list=[], data_type="resc"):
        if not infer_list:
            train_dataset, infer_dataset = self.get_dataset(data_type)
        else:
            if data_type == "oct":
                train_dataset = OCTDataset(
                    self.args, data_type="inference", infer_list=infer_list
                )
            elif data_type == "resc":
                train_dataset = RESCDataset(
                    self.args, data_type="inference", infer_list=infer_list
                )
            elif data_type == "duke":
                train_dataset = DukeDataset(
                    self.args, data_type="inference", infer_list=infer_list
                )
        dataloader = torch.utils.data.DataLoader(
            train_dataset, num_workers=8, batch_size=16, shuffle=False
        )

        for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image, labels, mask, shape, paths = (
                data["image"].to(self.args.device),
                data["labels"].to(self.args.device),
                data["mask"].to(self.args.device),
                data["shape"],
                data["path"]
            )
            img_names = [x.split("/")[-1].split(".")[0] for x in data["path"]]
            updated_image = image.clone()
            with torch.no_grad():
                gan_inputs = self.transform_norml(updated_image)
                healthy_img = self.gan_pretrained.inference(gan_inputs)
                norm_healthy_img = normalized_batch_tensor(healthy_img)    
                # tensor_for_att = diff_map_for_att(
                #     updated_image, healthy_img, mask
                # )
            for img, img_name, path in zip(norm_healthy_img, img_names, paths):
                # save healthy image
                # save_dir = get_gan_path_by_image_path(path)
                # save_name = ".".join(save_dir.split(".")[:-1]) + ".png"
                # ours 
                save_name = path.replace("original", "gan_healthy").split(".")[0] + ".png"
                # if image exists, skip
                if os.path.exists(save_name):
                    continue
                # import pdb; pdb.set_trace()
                vutils.save_image(
                    img.unsqueeze(0),
                    save_name,
                    normalize=True,
                    scale_each=False,
                )
            # import pdb; pdb.set_trace()

            # self.get_masks_and_save(img_name, data, resized_back_healthy_version)
            # edges = cv2.Canny(resized_back_healthy_version, 100, 200)


if __name__ == "__main__":
    is_inference = True
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dirs",
        type=str,
        default="datasets/our_dataset",
        help="root datasets directory: 2015_BOE_Chiu | RESC | our_dataset",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=["SRF", "IRF", "EZ disrupted", "HRD", "BackGround"],
        help="['SRF', 'IRF', 'EZ', 'HRD',  'RPE', 'BackGround', 'EZ attenuated', 'EZ disrupted', 'Retinal Traction', 'Definite DRIL']",
    )
    parser.add_argument(
        "--resc_labels",
        type=str,
        default=["SRF", "PED", "BackGround"],
        help="['SRF', 'PED', 'LESION', 'BackGround']",
    )
    parser.add_argument(
        "--boe_labels",
        type=str,
        default=["Fluid", "BackGround"],
        help="['Fluid', 'BackGround']",
    )
    parser.add_argument(
        "--expert_annot", type=str, default="both", help="mina, meera, both"
    )
    parser.add_argument(
        "--is_size",
        default=(512, 512),
        help="resize of input image, need same size as GANs generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )

    args = parser.parse_args()
    if "RESC" in args.root_dirs:
        args.mask_dir = "datasets/RESC/mask"
    elif "BOE" in args.root_dirs:
        args.annot_dir = "segment_annotation/labels"
    else:
        args.mask_dir = "datasets/our_dataset/mask"
    validator = Inference(args)
    start = time.time()
    # num_examples = validator.prepare_pesudo_label_for_seg()
    # validator.inference(infer_list=["sn22698_124.bmp"], data_type="resc")
    validator.inference(data_type="oct")
    # validator.inference(infer_list=['DR10.jpeg', 'DR91.jpeg', 'NORMAL-76914-1.jpeg', 'DME-3565572-7.jpeg', 'NORMAL-2709055-1.jpeg', 'DME-4240465-13.jpeg'], data_type='oct')

    # validator.inference(infer_list=['DME-15307-1.jpeg',
    #                               'DME-4240465-41.jpeg',
    #                               'DR10.jpeg',
    #                               'NORMAL-15307-1.jpeg'])

# %%
