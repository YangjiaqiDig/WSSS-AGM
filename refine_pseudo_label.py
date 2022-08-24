from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import numpy as np
from utils import OrgLabels, post_process_cam, out_cam_pred_alpha, get_num_classes
from pytorch_grad_cam import GradCAM
import torch.nn.functional as F

'''
5 labels: image -> 2 / 5 (irf, ez)  -> class_prob > 0.5
[0.1, 0.8, 0.1, 0.8, 1]
cam * 5 -> prob (0-1)
Muli(cam) -> 0
Sum(cam5) -> 1
filter cam5 -> cam2 -> Sum() -> normalize
'''
def refine_input_by_cam(device, multi_task_model, image, mask, aug_smooth=False):
    with torch.no_grad():
        outputs = multi_task_model(image)
    bacth_preds = (outputs > 0.5) * 1 # [batch, cls] -> (0,1)
    batch_cam_masks = []
    target_layers = [multi_task_model.module.base_model[-1][-1]] # .module. if use dataparallel
    with GradCAM(model=multi_task_model, use_cuda=device, target_layers=target_layers) as cam:
        for cls in range(len(OrgLabels)):
            targets = [ClassifierOutputTarget(cls)] * len(image) # for all in batch return the current class cam
            batch_grayscale_cam = cam(input_tensor=image,targets=targets, eigen_smooth=False, aug_smooth=aug_smooth)
            grayscale_tensor = torch.from_numpy(batch_grayscale_cam).to(device)
            grayscale_tensor = grayscale_tensor.unsqueeze(1).repeat(1, 3, 1, 1) # extend gray to 3 channels
            batch_cam_masks.append(grayscale_tensor) # cls, [batch, 3, w, h]
    updated_input_tensor = image.clone()
    for batch_idx, single_batch_pred in enumerate(bacth_preds):
        singel_cam_masks = [batch_cam_mask[batch_idx] for batch_cam_mask in batch_cam_masks] # cls, [3, w, h]
        # curr_preds = single_batch_pred # (cls) 0/1 values as outputs threshold by 0.5
        curr_preds = outputs[batch_idx] # classification probability
        '''if predict 1, keep the class cam, else 0, turn cam to 0 black image. Except last class BackGround'''
        target_classes_cam = [class_cam * curr_preds[l] for l, class_cam in enumerate(singel_cam_masks[:-1])]
        # sum the cams for predicted classes
        sum_masks = sum(target_classes_cam)
        # normalize the above 'attention map' to 0-1
        min, max = sum_masks.min(), sum_masks.max()
        sum_masks.add_(-min).div_(max - min + 1e-5) 
        ''' BackGround CAM * normalized CAM * Original Image. does norm -> multiply order matter? '''
        # background_mask = singel_cam_masks[-1] # Background CAM
        background_mask = mask[batch_idx].clone()
        background_mask[background_mask == 0] = 0.2
        soft_apply = sum_masks * background_mask # cam predicts * background constraints
        
        num_channels = 3#image.shape[1]
        for s in range(0, num_channels, 3):
            inputs_after_soft_addon = soft_apply * image[batch_idx, s:s+3,] # [3, w, h]
            # normilize the input image after addon soft map on origin input (both origin & gan)
            soft_min, soft_max = inputs_after_soft_addon.min(), inputs_after_soft_addon.max()
            inputs_after_soft_addon.add_(-soft_min).div_(soft_max - soft_min + 1e-5)
            updated_input_tensor[batch_idx, s:s+3,] = inputs_after_soft_addon
    return updated_input_tensor

def get_pseudo_label(params, multi_task_model):
    inputs, batch_preds, updated_image = params['inputs'], params['batch_preds'], params['refined']
    pseudo_labels = []
    target_layers = [multi_task_model.module.base_model[-1][-1]] # .module. if use dataparallel
    for i, pred in enumerate(batch_preds):
        orig_mask = inputs['mask'][i][0].clone()
        pred_classes = [i for i,v in enumerate(pred) if v > 0.5]        
        norm_cam_list = []
        for cls in pred_classes:
            if OrgLabels[cls] == 'BackGround':
                continue
            targets = [ClassifierOutputTarget(cls)]
            with GradCAM(model=multi_task_model, use_cuda=params['device'], target_layers=target_layers) as cam:
                grayscale_cam = cam(input_tensor=updated_image,targets=targets,eigen_smooth=False, aug_smooth=False)
            grayscale_cam = grayscale_cam[0, :]
            resized_cam = F.interpolate(torch.from_numpy(grayscale_cam).unsqueeze(0).unsqueeze(0), size=orig_mask.shape, mode='bilinear', align_corners=False)[0,0].numpy()
            single_norm_cam = post_process_cam(resized_cam, orig_mask)
            norm_cam_list.append(single_norm_cam)
        pred_with_bg_score = [np.ones_like(norm_cam_list[0]) * out_cam_pred_alpha] + norm_cam_list # (K + 1, [h, w])
        labels = np.argmax(np.array(pred_with_bg_score), axis=0) # [0 - num_class]
        pseudo_labels.append(labels)
    return torch.LongTensor(pseudo_labels)
