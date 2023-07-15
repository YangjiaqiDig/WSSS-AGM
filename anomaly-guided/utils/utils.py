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
    deprocess_image,
    preprocess_image,
)
import time
import torchvision.models as models
import torchvision.utils as vutils
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
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)  # [0]


class CAMGeneratorAndSave:
    def __init__(self, opts, multi_task_model, epoch=None) -> None:
        self.opts = opts
        self.multi_task_model = multi_task_model
        self.model = SegmentationModelOutputWrapper(self.multi_task_model)
        self.epoch = epoch
        non_background_names = [x for x in OrgLabels if "BackGround" != x]
        self.lesion_classes = [OrgLabels.index(name) for name in non_background_names]
        self.save_results = self.opts.save_cam_images

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_cam_results_per_class(self, orig_img, orig_mask, norm_annot):
        save_cam_in_row = []
        rgb_img = (orig_img / 255).copy()
        if rgb_img.ndim == 2:
            rgb_img = np.repeat(rgb_img[..., np.newaxis], 3, -1)

        target_layers = self.multi_task_model.get_cam_target_layers()
        extend_input_tensor = self.updated_image.repeat(
            get_num_classes(), 1, 1, 1
        )  # [1, 9, 512, 512] -> [2, 9, 512, 512]
        targets = [
            ClassifierOutputTarget(cls)
            for cls in range(len(OrgLabels))
            if OrgLabels[cls] != "BackGround"
        ]
        # print(self.model)
        # startt = time.time()
        with GradCAM(
            model=self.model,
            use_cuda="cuda",
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        ) as cam:
            # import pdb; pdb.set_trace()
            grayscale_cam = cam(
                input_tensor=extend_input_tensor,
                targets=targets,
                eigen_smooth=False,
                aug_smooth=self.opts.aug_smooth,
            )
            # print(grayscale_cam)
            # [cls, w, h]
        # print(time.time()-startt)

        resized_cam = F.interpolate(
            torch.from_numpy(grayscale_cam).unsqueeze(0),  # [1, cls, w, h]
            size=orig_mask.shape,
            mode="bilinear",
            align_corners=False,
        )[
            0
        ].numpy()  # [cls, w, h] numpy

        norm_resized_cam = post_process_cam(resized_cam, orig_mask)
        # '''save the prob heat map'''
        # gt_lesions = np.unique(norm_annot[norm_annot!=0])
        # save_file = {i_c-1: norm_resized_cam[i_c-1] for i_c in gt_lesions}
        # n = self.img_name.split('.')[0]
        # # np.save(f'paper/abl-heads/8/{n}.npy', save_file)
        # np.save(f'datasets/our_dataset/pseudo_map/{n}.npy', save_file)
        # '''end'''
        
        # import pdb; pdb.set_trace()
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
        save_img = [orig_img, orig_mask, orig_annot]
        save_image_h = cv2.hconcat(save_img)

        if self.is_inference:
            save_path = os.path.join(
                self.opts.save_inference, "{}".format(self.img_name.split(".")[0])
            )
        else:
            save_path = os.path.join(
                self.opts.save_folder,
                "iteration",
                "{}".format(self.img_name.split(".")[0]),
            )
        if self.save_results:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path + "/orig_{}.jpg".format(truth_label), save_image_h)

            if not self.is_inference:
                w, h = self.opts.is_size[0], self.opts.is_size[1]
                save_updated_img = self.updated_image[batch_nb].reshape(-1, 3, w, h)
                vutils.save_image(
                    save_updated_img,
                    save_path
                    + "/epoch{0}_refined_{1}.jpg".format(self.epoch, truth_label),
                    normalize=True,
                    scale_each=True,
                )
        norm_annot = convert_data_labels(orig_annot, self.opts.root_dirs)
        save_cam_in_row, pred_labels = self.get_cam_results_per_class(
            orig_img, orig_mask, norm_annot
        )

        return norm_annot, pred_labels, save_cam_in_row, save_path

    def get_cam_and_save(self, params, is_inference=False):
        # batch_preds [BC] B: batch, C: Class
        self.inputs, self.updated_image, batch_preds = (
            params["inputs"],
            params["refined"],
            params["batch_preds"],
        )
        ready_pred_4d = []
        gt = []
        self.is_inference = is_inference
        for batch_nb, single_pred in enumerate(batch_preds):
            ground_true_classes = [
                i for i, v in enumerate(self.inputs["labels"][batch_nb]) if v > 0.5
            ]
            # only calculate and save for ground truth lesion images
            if not len(set(self.lesion_classes) & set(ground_true_classes)):
                continue
            gt_labels, pred_labels, save_cam_in_row, save_path = self.save_cam_process(
                batch_nb
            )
            gt.append(gt_labels)
            ready_pred_4d.append(pred_labels)
            cv2.imwrite('datasets/RESC/pseudo_label_2/{}.png'.format(self.img_name.split('.')[0]), pred_labels)
            # import pdb; pdb.set_trace()
            if not self.save_results:
                continue
            single_pred_cls = [i for i, v in enumerate(single_pred) if v > 0.5]
            save_class_name = "_".join(
                [
                    OrgLabels[cls]
                    for cls in single_pred_cls
                    if OrgLabels[cls] != "BackGround"
                ]
            )
            # for paper
            # for i, each in enumerate(save_cam_in_row):
            #     cv2.imwrite(
            #         'paper/refine-illu' + "/{}.png".format(i), np.array(each)
            #     )
            if len(save_cam_in_row):
                im_h = cv2.hconcat(save_cam_in_row)
                if self.is_inference:
                    cv2.imwrite(
                        save_path + "/pred_{0}.jpg".format(save_class_name), im_h
                    )
                else:
                    cv2.imwrite(
                        save_path
                        + "/epoch{0}_{1}.jpg".format(self.epoch, save_class_name),
                        im_h,
                    )
        return gt, ready_pred_4d

    def get_pseudo_labels(self, params):
        self.inputs, self.updated_image = (params["inputs"], params["refined"])
        ready_pred_4d = []
        gt = []
        for batch_nb, sing_labels in enumerate(self.inputs["labels"]):
            ground_true_classes = [i for i, v in enumerate(sing_labels) if v > 0.5]
            # only calculate and save for ground truth lesion images
            if not len(set(self.lesion_classes) & set(ground_true_classes)):
                continue
            self.img_name = self.inputs["path"][batch_nb].split("/")[-1]
            # 0-255 pixel value, numpy
            image_path = self.inputs["path"][batch_nb]
            orig_mask = load_mask(image_path)
            # norm_annot = get_resc_train_annot(self.img_name)

            target_layers = self.multi_task_model.get_cam_target_layers()
            extend_input_tensor = self.updated_image.repeat(
                get_num_classes(), 1, 1, 1
            )  # [1, 9, 512, 512] -> [2, 9, 512, 512]
            targets = [
                ClassifierOutputTarget(cls)
                for cls in range(len(OrgLabels))
                if OrgLabels[cls] != "BackGround"
            ]
            with GradCAM(
                model=self.model,
                use_cuda="cuda",
                target_layers=target_layers,
                reshape_transform=reshape_transform,
            ) as cam:
                grayscale_cam = cam(
                    input_tensor=extend_input_tensor,
                    targets=targets,
                    eigen_smooth=False,
                    aug_smooth=True,
                )
            resized_cam = F.interpolate(
                torch.from_numpy(grayscale_cam).unsqueeze(0),  # [1, cls, w, h]
                size=orig_mask.shape,
                mode="bilinear",
                align_corners=False,
            )[
                0
            ].numpy()  # [cls, w, h] numpy

            norm_resized_cam = post_process_cam(resized_cam, orig_mask)
            """Generate all cams except the ones that are not in ground true classes"""
            for i in range(get_num_classes()):
                if i not in ground_true_classes:
                    norm_resized_cam[i] = 0
                    continue
            # import pdb; pdb.set_trace()
            img_save_name = self.img_name.split('.')[0]
            #  for duke
            # img_save_name = (
            #     self.inputs["path"][batch_nb].replace("/", "+").rsplit(".", 1)[0]
            # )
            # np.save(f'abl_backbone_maps/resc_resnet18/{img_save_name}.npy', norm_resized_cam)
            # import pdb; pdb.set_trace()
            bg_score = [
                np.ones_like(norm_resized_cam[0]) * self.opts.out_cam_pred_alpha
            ]
            pred_with_bg_score = np.concatenate((bg_score, norm_resized_cam))

            pred_labels = np.argmax(pred_with_bg_score, axis=0)  # [0 - num_class]
            # import pdb; pdb.set_trace()
            save_pseudo_label_path = os.path.join(
                self.opts.root_dirs, "pseudo_label_2", img_save_name + ".png"
            )
            cv2.imwrite(save_pseudo_label_path, pred_labels)

            # gt.append(norm_annot)
            ready_pred_4d.append(pred_labels)
        return gt, ready_pred_4d


def save_models(
    args, epoch, multi_task_model, multi_optimizer, best_type=None, is_iter=False
):
    save_path = f"./{args.save_folder}/weights"
    if best_type is None:
        save_name = epoch + 1
    elif best_type == "pseudo":
        save_name = "best_iou"
    elif best_type == "cls":
        save_name = "best"
    else:
        raise ValueError("The type for save model is not available")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if is_iter:
        save_name = f"{is_iter}_{save_name}"
    torch.save(
        {
            "epoch": epoch,
            "args": args,
            "state_dict": multi_task_model.state_dict(),
            "optimizer": multi_optimizer.state_dict(),
        },
        save_path + "/{0}.pwf".format(save_name),
    )


def save_tensorboard_val(tb, loss_dict, mark_epoch):
    tb.add_scalar("Loss/Valid", loss_dict["total_val_loss"], mark_epoch + 1)
    tb.add_scalar("ROC/Valid", loss_dict["total_val_roc"], mark_epoch + 1)
    for acc_type in ["acc", "f1m"]:
        tb.add_scalar(
            "Val Accuracy/{}".format(acc_type),
            loss_dict["total_val_acc_matrix"][acc_type],
            mark_epoch + 1,
        )
    for label_type in OrgLabels:
        tb.add_scalar(
            "Val Class Acc/{}".format(label_type),
            loss_dict["total_val_acc_matrix"][label_type],
            mark_epoch + 1,
        )
        tb.add_scalar(
            "Val Class ROC/{}".format(label_type),
            loss_dict["total_val_roc_matrix"][label_type],
            mark_epoch + 1,
        )
    tb.close()


def save_tensorboard_train(tb, loss_dict, mark_epoch):
    tb.add_scalar("Loss/Train", loss_dict["total_train_loss"], mark_epoch + 1)
    tb.add_scalar("ROC/Train", loss_dict["total_train_roc"], mark_epoch + 1)
    for acc_type in ["acc", "f1m"]:
        tb.add_scalar(
            "Train Accuracy/{}".format(acc_type),
            loss_dict["total_train_acc_matrix"][acc_type],
            mark_epoch + 1,
        )
    for label_type in OrgLabels:
        tb.add_scalar(
            "Train Class Acc/{}".format(label_type),
            loss_dict["total_train_acc_matrix"][label_type],
            mark_epoch + 1,
        )
        tb.add_scalar(
            "Train Class ROC/{}".format(label_type),
            loss_dict["total_train_roc_matrix"][label_type],
            mark_epoch + 1,
        )
    tb.close()


if __name__ == "__main__":
    backbone = models.resnet18(pretrained=True)
    num_class = len(OrgLabels)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "fullgrad": FullGrad,
    }

    model = models.resnet50(pretrained=True)
