from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import numpy as np
from utils.utils import (
    OrgLabels,
    SegmentationModelOutputWrapper,
    post_process_cam,
    get_num_classes,
)
from pytorch_grad_cam import GradCAM
import torch.nn.functional as F

"""
5 labels: image -> 2 / 5 (irf, ez)  -> class_prob > 0.5
[0.1, 0.8, 0.1, 0.8, 1]
cam * 5 -> prob (0-1)
Muli(cam) -> 0
Sum(cam5) -> 1
filter cam5 -> cam2 -> Sum() -> normalize
"""


def refine_input_by_cam(
    device, multi_task_model, input_tensor, mask_tensor, aug_smooth=False
):
    multi_task_model.eval()
    with torch.no_grad():
        cls_outputs = multi_task_model(input_tensor)
    batch_cam_masks = []
    target_layers = multi_task_model.get_cam_target_layers()
    wrap_model = SegmentationModelOutputWrapper(multi_task_model)
    # TODO: replace the range with 'BackGround' label not the last position case
    for cls in range(get_num_classes()):
        targets = [ClassifierOutputTarget(cls)] * len(
            input_tensor
        )  # for all in batch return the current class cam
        with GradCAM(
            model=wrap_model, use_cuda=device, target_layers=target_layers
        ) as cam:
            batch_grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
                eigen_smooth=False,
                aug_smooth=aug_smooth,
            )
        batch_cam_masks.append(batch_grayscale_cam)  # cls, [batch, w, h]
    batch_cam_masks = np.transpose(
        np.stack(batch_cam_masks), (1, 0, 2, 3)
    )  # np: [batch, cls, w, h]

    updated_input_tensor = input_tensor.clone()
    for batch_idx, singel_cam_masks in enumerate(batch_cam_masks):
        # curr_preds = single_batch_pred # (cls) 0/1 values as cls_outputs threshold by 0.5
        curr_preds = cls_outputs[batch_idx]  # classification probability
        norm_cams = torch.from_numpy(post_process_cam(singel_cam_masks)).to(device)
        """CAM heat map * class probability"""
        target_classes_cam = [
            class_cam * curr_preds[cls_i] for cls_i, class_cam in enumerate(norm_cams)
        ]
        # sum the cams for predicted classes
        sum_masks = sum(target_classes_cam)  # (w, h)
        # normalize the above 'attention map' to 0-1
        min, max = sum_masks.min(), sum_masks.max()
        sum_masks.add_(-min).div_(max - min + 1e-5)
        soft_apply = sum_masks.unsqueeze(0).repeat(3, 1, 1)  # (3, w, h)
        """ BackGround CAM * normalized CAM * Original Image. does norm -> multiply order matter? """
        num_channels = 3  # image.shape[1]
        for s in range(0, num_channels, 3):
            inputs_after_soft_addon = (
                soft_apply
                * input_tensor[
                    batch_idx,
                    s : s + 3,
                ]
            )  # [3, w, h]
            # normilize the input image after addon soft map on origin input (both origin & gan)
            soft_min, soft_max = (
                inputs_after_soft_addon.min(),
                inputs_after_soft_addon.max(),
            )
            inputs_after_soft_addon.add_(-soft_min).div_(soft_max - soft_min + 1e-5)
            updated_input_tensor[
                batch_idx,
                s : s + 3,
            ] = inputs_after_soft_addon
        # import torchvision.utils as vutils
        # vutils.save_image(updated_input_tensor.reshape(3,3,512,512), 'test.png', normalize=True, scale_each=True)
    return updated_input_tensor


def get_pseudo_label(params, multi_task_model):
    retinal_mask, input_tensor, cls_labels, args = (
        params["mask"].cpu().numpy(),
        params["input_tensor"],
        params["cls_labels"].cpu().numpy(),
        params["args"],
    )
    pseudo_labels = []
    batch_cam_masks = []
    target_layers = (
        multi_task_model.get_cam_target_layers()
    )  # .module. if use dataparallel
    wrap_model = SegmentationModelOutputWrapper(multi_task_model)

    for cls in range(get_num_classes()):
        targets = [ClassifierOutputTarget(cls)] * len(
            input_tensor
        )  # for all in batch return the current class cam
        with GradCAM(
            model=wrap_model, use_cuda=args.device, target_layers=target_layers
        ) as cam:
            batch_grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
                eigen_smooth=False,
                aug_smooth=args.aug_smooth,
            )
        batch_cam_masks.append(batch_grayscale_cam)  # cls, [batch, w, h]
    batch_cam_masks = np.transpose(
        np.stack(batch_cam_masks), (1, 0, 2, 3)
    )  # np: [batch, cls, w, h]

    for singel_cam_masks, single_retinal_mask, cls_label in zip(
        batch_cam_masks, retinal_mask, cls_labels
    ):
        norm_cams = post_process_cam(
            singel_cam_masks, single_retinal_mask[0]
        )  # [cls, w, h]
        for i in range(get_num_classes()):
            if cls_label[i] == 0:
                norm_cams[i] = 0
        bg_score = [np.ones_like(norm_cams[0]) * args.out_cam_pred_alpha]
        pred_with_bg_score = np.concatenate((bg_score, norm_cams))  # [cls+1, w, h]
        """Generate psuedo label by gt labels"""
        pred_labels = np.argmax(pred_with_bg_score, axis=0)  # [0 - num_class]
        pseudo_labels.append(pred_labels)
    import pdb

    pdb.set_trace()
    return torch.LongTensor(pseudo_labels)
