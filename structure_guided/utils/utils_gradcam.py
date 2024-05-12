from math import sqrt
import os
import torch
import numpy as np
from options import Configs
from pytorch_grad_cam import (
    GradCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
    GuidedBackpropReLUModel,
)
import cv2
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,
)
import torchvision.models as models
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F
from PIL import Image

type_color = {
    0: [0, 0, 0],  # black
    1: [52, 195, 235],  # srf red
    3: [0, 255, 0],  # irf green
    2: [165, 235, 52],  # ez blue   235, 211, 52
    4: [255, 255, 0],  # hrd yellow
    5: [255, 165, 0],  # rpe orange
    6: [255, 0, 255],  # back ground pink
}
OrgLabels = Configs().get_labels()  # ['SRF', 'IRF', 'EZ', 'HRD',  'RPE', 'BackGround']


CAT_LIST = ["background"] + [x for x in OrgLabels if x != "BackGround"] + ["meanIOU"]


def get_mask_path_by_image_path(image_path):
    if "BOE" in image_path:
        mask_path = image_path.replace("/images/", "/mask/")
    elif "RESC" in image_path:
        mask_path = image_path.replace("/RESC/", "/RESC/mask/").replace(
            "original_images/", ""
        )
    elif "NORMAL" in image_path:
        mask_path = image_path.replace("train/0.normal", "normal_mask")
    else:
        mask_path = image_path.replace("original", "mask")
    return mask_path


def load_mask(image_path):
    orig_mask = np.asarray(Image.open(get_mask_path_by_image_path(image_path)))[..., 0]
    orig_mask[orig_mask > 150] = 255
    orig_mask[orig_mask <= 150] = 0
    return orig_mask


def get_annot_by_dataset(opts, img_name):
    if "our_dataset" in opts.root_dirs:
        if opts.expert_annot == "both":
            annot_path = os.path.join(
                opts.root_dirs, opts.annot_dir, img_name.split(".")[0] + ".png"
            )
        else:
            expert_annot_img_name = img_name.split(".")[0] + f"_{opts.expert_annot}.png"
            annot_path = os.path.join(
                opts.root_dirs, opts.annot_dir, expert_annot_img_name
            )
    else:
        annot_path = os.path.join(
            opts.root_dirs, opts.annot_dir, img_name
        )  # resc valid/label_images
    orig_annot = np.asarray(Image.open(annot_path))
    return orig_annot


def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_swim(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_segformer(tensor):
    if tensor.shape[1] == 512:
        input_t = tensor[:, :256, :]
    else:
        input_t = tensor
    height = width = int(sqrt(input_t.shape[1]))
    result = input_t.reshape(input_t.size(0), height, width, input_t.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform(tensor):
    if tensor.shape[1] != 197 and tensor.shape[1] != 49:
        return tensor
    if tensor.shape[1] == 197:
        return reshape_transform_vit(tensor)
    return reshape_transform_swim(tensor)


def get_num_classes():
    if "BackGround" in OrgLabels:
        return len(OrgLabels) - 1
    return len(OrgLabels)


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


# input is array H * W
def convert_resc_pix2image_label(label_pixel_arr):
    # back: 0, ped: 128, srf: 191, retinal: 255
    image_label = {"SRF": 0, "PED": 0, "lesion": 0, "health": 0, "BackGround": 1}
    labels = np.unique(label_pixel_arr)
    if len(labels) == 1:
        image_label["health"] += 1
    if 128 in labels:
        image_label["PED"] += 1
    if 191 in labels:
        image_label["SRF"] += 1
    if 255 in labels:
        image_label["lesion"] += 1
    return image_label


def convert_resc_labels(img):
    # 0 background,
    # 1 lesion(need turn to background),  1 -> 0
    # 0.74 SRF(need turn to 1),           0.74 -> 1
    # 0.51 PED(need turn to 2)            0.51 -> 2
    # back: 0, ped: 128, srf: 191, retinal: 255
    img[img == 255] = 0
    img[img == 191] = 1
    img[img == 128] = 2
    return img


def convert_duke_labels(img):
    img[img < 255] = 0
    img[img == 255] = 1
    return img


def convert_our_dataset_labels(img):
    # "categories":[{"id":51,"name":"IRF"},{"id":102,"name":"SRF"},{"id":153,"name":"HRD"},{"id":204,"name":"EZ disruption"},{"id":255,"name":"RPE "}]}
    # ['SRF', 'IRF', 'EZ disrupted', 'HRD', 'BackGround']
    img[img == 51] = 2
    img[img == 102] = 1
    img[img == 153] = 4
    img[img == 204] = 3
    img[img == 255] = 0  # we dont include RPE for now

    return img


def convert_data_labels(img, root_dirs):
    norm_img = img.copy()
    if "RESC" in root_dirs:
        return convert_resc_labels(norm_img)
    if "BOE" in root_dirs:
        return convert_duke_labels(norm_img)
    return convert_our_dataset_labels(norm_img)


def get_resc_train_annot(img_name):
    annot_path = os.path.join("datasets/RESC/train/label_images", img_name)
    orig_annot = np.asarray(Image.open(annot_path))
    return convert_resc_labels(orig_annot)


def post_process_cam(resized_cam, orig_mask=None):
    masked_cam = resized_cam
    if orig_mask is not None:
        mask_clone = orig_mask.copy()  # NEUROSENSORY RETINA only (ILM to RPE)
        mask_clone[mask_clone == 0] = 0.5
        masked_cam = resized_cam * mask_clone

    # cam_max = masked_cam.max()
    # cam_min = masked_cam.min()
    cam_max = np.max(masked_cam, (1, 2), keepdims=True)
    cam_min = np.min(masked_cam, (1, 2), keepdims=True)
    masked_cam[masked_cam < cam_min + 1e-5] = 0

    norm_masked_cam = (masked_cam - cam_min - 1e-5) / (cam_max - cam_min + 1e-5)
    return norm_masked_cam


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model, branch):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model
        self.branch = branch

    def forward(self, x):
        outputs = self.model(x)
        if self.branch == "x":
            return outputs[0]["main-cls"]
        return outputs[1]["layer-cls"]


class CAMGeneratorAndSave:
    def __init__(self, opts, multi_task_model, epoch=None) -> None:
        self.opts = opts
        self.multi_task_model = multi_task_model
        self.modelx = SegmentationModelOutputWrapper(self.multi_task_model, branch="x")
        # self.modely = SegmentationModelOutputWrapper(self.multi_task_model, branch="y")
        self.epoch = epoch
        non_background_names = [x for x in OrgLabels if "BackGround" != x]
        self.lesion_classes = [OrgLabels.index(name) for name in non_background_names]
        self.save_results = self.opts.save_results

    def operate_gradcam(self, rescale_size, batch_nb):
        target_layers_x = self.multi_task_model.get_cam_target_layers("x")
        input_to_gradcam = self.input_tensor[batch_nb].unsqueeze(0)
        extend_input_tensor = input_to_gradcam.repeat(
            get_num_classes(), 1, 1, 1
        )  # [1, 9, 512, 512] -> [2, 9, 512, 512]
        targetsx = [
            ClassifierOutputTarget(cls)
            for cls in range(len(OrgLabels))
            if OrgLabels[cls] != "BackGround"
        ]
        with GradCAM(
            model=self.modelx,
            use_cuda="cuda",
            target_layers=target_layers_x,
            reshape_transform=reshape_segformer,
        ) as cam:
            grayscale_cam_x = cam(
                input_tensor=extend_input_tensor,
                targets=targetsx,
                eigen_smooth=False,
                # aug_smooth=self.opts.aug_smooth,
            )
        grayscale_cam = grayscale_cam_x  # + grayscale_cam_y
        # normalize
        # grayscale_cam = grayscale_cam / np.max(grayscale_cam, (1, 2), keepdims=True)
        cam_max = np.max(grayscale_cam, (1, 2), keepdims=True)
        cam_min = np.min(grayscale_cam, (1, 2), keepdims=True)
        grayscale_cam = (grayscale_cam - cam_min) / (cam_max - cam_min + 1e-7)
        # if 1 in norm_annot:
        #     cv2.imwrite(f"temp_save/1_{self.img_name}", grayscale_cam[0] * 255)
        # if 2 in norm_annot:
        #     cv2.imwrite(f"temp_save/2_{self.img_name}", grayscale_cam[1] * 255)
        # import pdb; pdb.set_trace()

        resized_cam = F.interpolate(
            torch.from_numpy(grayscale_cam).unsqueeze(0),  # [1, cls, w, h]
            size=rescale_size,
            mode="bilinear",
            align_corners=False,
        )[
            0
        ].numpy()  # [cls, w, h] numpy

        return resized_cam


    def get_cam_results_per_class(self, orig_img, orig_mask, norm_annot, batch_nb):
        save_cam_in_row = []
        rgb_img = (orig_img / 255).copy()
        if rgb_img.ndim == 2:
            rgb_img = np.repeat(rgb_img[..., np.newaxis], 3, -1)

        resized_cam = self.operate_gradcam(orig_mask.shape, batch_nb)
        norm_resized_cam = post_process_cam(resized_cam, orig_mask)

        bg_score = [np.ones_like(norm_resized_cam[0]) * self.opts.out_cam_pred_alpha]
        pred_with_bg_score = np.concatenate((bg_score, norm_resized_cam))

        """Generate all cams except the ones that are not in ground true classes"""
        # Need include the 0 == background, add 1 extra.
        for i in range(get_num_classes() + 1):
            if i not in norm_annot:
                pred_with_bg_score[i] = 0
                continue
            # we dont need save the background cam
            if i != 0 and self.save_results:
                visualization = show_cam_on_image(
                    rgb_img, pred_with_bg_score[i], use_rgb=True
                )
                cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
                save_cam_in_row.append(cam_image)

        pred_labels = np.argmax(pred_with_bg_score, axis=0)  # [0 - num_class]

        color_mask = np.zeros_like(rgb_img)
        for i_cls in range(1, get_num_classes() + 1):
            mask = pred_labels == i_cls
            color_mask[:, :,][
                mask
            ] = type_color[i_cls]
        color_mask = cv2.cvtColor(color_mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
        save_cam_in_row.append(color_mask)
        dst = cv2.addWeighted(
            (rgb_img * 255).astype(np.uint8), 1, color_mask.astype(np.uint8), 0.7, 0
        )
        save_cam_in_row.append(dst)

        return save_cam_in_row, pred_labels

    def save_cam_process(self, batch_nb):
        self.img_name = self.inputs["path"][batch_nb].split("/")[-1]
        # 0-255 pixel value, numpy
        image_path = self.inputs["path"][batch_nb]
        orig_img = np.asarray(Image.open(image_path))
        if len(orig_img.shape) == 3:
            orig_img = orig_img[..., 0]
        orig_mask = load_mask(image_path)
        orig_annot = get_annot_by_dataset(self.opts, self.img_name)

        true_classes = [
            i for i, v in enumerate(self.inputs["labels"][batch_nb]) if v > 0.5
        ]
        truth_label = [OrgLabels[cls] for cls in true_classes]
        truth_label = "_".join(truth_label)

        norm_annot = convert_data_labels(orig_annot, self.opts.root_dirs)
        save_cam_in_row, pred_labels = self.get_cam_results_per_class(
            orig_img, orig_mask, norm_annot, batch_nb
        )

        return norm_annot, pred_labels, save_cam_in_row

    def get_cam_and_save(self, params):
        # batch_preds [BC] B: batch, C: Class
        self.inputs, self.input_tensor = (
            params["inputs"],
            params["input_tensor"],
        )
        ready_pred_4d = []
        gt = []
        binary_labels = self.inputs["binary_labels"]
        for batch_nb in range(len(self.input_tensor)):
            # only calculate and save for ground truth lesion images
            if binary_labels[batch_nb] == 0:
                continue
            gt_labels, pred_labels, save_cam_in_row = self.save_cam_process(batch_nb)
            gt.append(gt_labels)
            ready_pred_4d.append(pred_labels)

            if not self.save_results:
                continue
            
            ground_true_classes = [
                i for i, v in enumerate(self.inputs["labels"][batch_nb]) if v > 0.5
            ]
            save_class_name = "_".join(
                [
                    OrgLabels[cls]
                    for cls in ground_true_classes
                    if OrgLabels[cls] != "BackGround"
                ]
            )
            save_path = os.path.join(
                self.opts.save_folder, "images/{}".format(self.img_name.split(".")[0])
            )
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            if len(save_cam_in_row):
                im_h = cv2.hconcat(save_cam_in_row)
                cv2.imwrite(
                    save_path + "/epoch{0}_{1}.jpg".format(self.epoch, save_class_name),
                    im_h,
                )
        return gt, ready_pred_4d
