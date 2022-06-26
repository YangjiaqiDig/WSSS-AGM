from preprocess import background_mask
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import os
import numpy as np
import torchvision.models as models
from utils import OrgLabels, type_color
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
import torchvision.utils as vutils


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def save_cam_during_train(params, cam):
    args, epoch, inputs, batch_preds, updated_image = params['args'], params['epoch'], params['inputs'], params['batch_preds'], params['refined']
    for i, pred in enumerate(batch_preds):
        orig_image = inputs["image"][i][0].clone()
        orig_mask = inputs['mask'][i][0].clone()
        rgb_img = (np.float32(inputs["image"][i][:3].permute(1, 2, 0)))
        img_path = inputs["path"][i].split('/')[-1]
        save_path = os.path.join(args.save_folder, 'iteration', '{}'.format(img_path.split('.')[0]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
        # truth_label = '{0}_{1}'.format(inputs["labels"][i][0], inputs["labels"][i][1])
        true_classes = [i for i,v in enumerate(inputs["labels"][i]) if v > 0.5]
        truth_label = [OrgLabels[cls] for cls in true_classes]
        truth_label = '_'.join(truth_label)
        w, h = inputs["image"][i].shape[-2], inputs["image"][i].shape[-1]
        save_img = inputs["image"][i].reshape(-1,3,w, h)
        save_updated_img = updated_image[i].reshape(-1,3,w, h)
        vutils.save_image(save_img, save_path + '/orig_{}.jpg'.format(truth_label), normalize=True, scale_each=True) # scale_each limit normalize for each independently in batch
        vutils.save_image(save_updated_img, save_path + '/epoch{0}_refined_{1}.jpg'.format(epoch, truth_label), normalize=True, scale_each=True)
        
        pred_classes = [i for i,v in enumerate(pred) if v > 0.5]
        save_cam_in_row = []
        save_class_name = ''
        
        all_grey = [np.zeros((w, h))] * (len(OrgLabels) + 1)
        for cls in pred_classes:
            targets = [ClassifierOutputTarget(cls)]
            # input_tensor  = inputs["image"].to(args.device)
            grayscale_cam = cam(input_tensor=updated_image,targets=targets,eigen_smooth=False, aug_smooth=True)
            grayscale_cam = grayscale_cam[0, :]
            grey_thred = grayscale_cam.copy()
            grey_thred[grey_thred < 0.3] = 0
            
            strict_grey = grey_thred.copy()
            if OrgLabels[cls] in ['IRF', 'SRF']:
                strict_grey[(orig_image > 0.2) | (orig_mask == 0)] = 0 #keep bubble region (dark)
            else: strict_grey[(orig_image < 0.5) | (orig_mask == 0)] = 0 # hrd, ez keep the light layer region
                     
            if OrgLabels[cls] != 'BackGround': # dont illustrate the background
                all_grey[cls + 1] = strict_grey
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            save_cam_in_row.append(cam_image)
            save_class_name =  save_class_name + '_' + OrgLabels[cls]
        
        labels = np.argmax(np.array(all_grey), axis=0)
        color_mask = np.zeros((w, h, 3))
        # there is no leision, so no need show black image
        if labels.max() != 0:
            for i in range(1, len(OrgLabels) + 1):
                mask = labels == i
                color_mask[:,:,][mask] = type_color[i]
            color_mask = cv2.cvtColor(color_mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
            save_cam_in_row.append(color_mask)
        dst = cv2.addWeighted((rgb_img * 255).astype(np.uint8), 1, color_mask.astype(np.uint8), 0.7, 0)
        save_cam_in_row.append(dst)
        
        if (len(save_cam_in_row)):
            im_h = cv2.hconcat(save_cam_in_row)
            try:
                cv2.imwrite(save_path + '/epoch{0}_{1}.jpg'.format(epoch, save_class_name), im_h)
            except:
                print(save_class_name, im_h)
                import pdb; pdb.set_trace()

def save_cam_for_inference(params, cam):
    args, inputs, batch_preds, updated_image = params['args'], params['inputs'], params['batch_preds'], params['refined']
    is_background_include = 'BackGround'in OrgLabels
    # softmax = nn.Softmax()
    for i, pred in enumerate(batch_preds):
        orig_image = inputs["image"][i][0].clone()
        orig_mask = inputs['mask'][i][0].clone()
        rgb_img = (np.float32(inputs["image"][i][:3].permute(1, 2, 0)))
        img_path = inputs["path"][i].split('/')[-1]
        save_path = os.path.join(args.save_inference, '{}'.format(img_path.split('.')[0]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
        true_classes = [i for i,v in enumerate(inputs["labels"][i]) if v > 0.5]
        
        truth_label = [OrgLabels[cls] for cls in true_classes]
        truth_label = '_'.join(truth_label)
        w, h = inputs["image"][i].shape[-2], inputs["image"][i].shape[-1]
        save_img = updated_image[i].reshape(-1,3,w, h)
        vutils.save_image(save_img, save_path + '/orig_{}.jpg'.format(truth_label), normalize=True, scale_each=True)

        pred_classes = [i for i,v in enumerate(pred) if v > 0.5]
        if is_background_include and len(pred_classes)==1 and OrgLabels.index('BackGround') == true_classes[0]:
            continue
        save_cam_in_row = []
        save_class_name = ''
        
        all_grey = [np.zeros((w, h))] * (len(OrgLabels) + 1)
        for cls in pred_classes:
            targets = [ClassifierOutputTarget(cls)]
            grayscale_cam = cam(input_tensor=updated_image,targets=targets,eigen_smooth=False, aug_smooth=True)
            grayscale_cam = grayscale_cam[0, :]
            grey_thred = grayscale_cam.copy()
            grey_thred[grey_thred < 0.3] = 0
            
            strict_grey = grey_thred.copy()
            if OrgLabels[cls] in ['IRF', 'SRF']:
                strict_grey[(orig_image > 0.2) | (orig_mask == 0)] = 0 #keep bubble region (dark)
                # strict_grey = strict_grey * (1 - orig_image).cpu().numpy()
            else: strict_grey[(orig_image < 0.5) | (orig_mask == 0)] = 0 # hrd, ez keep the light layer region
            # strict_grey[strict_grey < 0.7] = 0
            
            # cv2.imwrite('test{0}.jpg'.format(cls), strict_grey * 255)
            if OrgLabels[cls] != 'BackGround':
                all_grey[cls + 1] = strict_grey
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            save_cam_in_row.append(cam_image)
            save_class_name =  save_class_name + '_' + OrgLabels[cls]

        labels = np.argmax(np.array(all_grey), axis=0)
        color_mask = np.zeros((w, h, 3))
        for i in range(1, len(OrgLabels) + 1):
            mask = labels == i
            color_mask[:,:,][mask] = type_color[i]
        color_mask = cv2.cvtColor(color_mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
        save_cam_in_row.append(color_mask)
        
        dst = cv2.addWeighted((rgb_img * 255).astype(np.uint8), 1, color_mask.astype(np.uint8), 0.7, 0)
        # overlap_mask = cv2.cvtColor(color_mask.astype(np.uint8), cv2.COLOR_RGB2BGR)
        save_cam_in_row.append(dst)
        
        if (len(save_cam_in_row)):
            im_h = cv2.hconcat(save_cam_in_row)
            try:
                cv2.imwrite(save_path + '/{0}.jpg'.format(save_class_name), im_h)
            except:
                print(save_class_name, im_h)
                import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
               
def save_cam_results(params, is_inference=False):
    cam = params['cam']
    # batch_preds [BC] B: batch, C: Class
    if is_inference:
        save_cam_for_inference(params, cam)
        return
    save_cam_during_train(params, cam)

'''
5 labels: image -> 2 / 5 (irf, ez)  -> class_prob > 0.5
[0.1, 0.8, 0.1, 0.8, 1]
cam * 5 -> prob (0-1)
Muli(cam) -> 0
Sum(cam5) -> 1
filter cam5 -> cam2 -> Sum() -> normalize
'''
def refine_input_by_cam(args, model, image, mask, cam, aug_smooth=True):
    outputs = model(image)
    bacth_preds = (outputs > 0.5) * 1 # [batch, cls] -> (0,1)
    batch_cam_masks = []
    for cls in range(len(OrgLabels)):
        targets = [ClassifierOutputTarget(cls)] * len(image) # for all in batch return the current class cam
        batch_grayscale_cam = cam(input_tensor=image,targets=targets,eigen_smooth=False, aug_smooth=aug_smooth)
        grayscale_tensor = torch.from_numpy(batch_grayscale_cam).to(args.device)
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
        # BackGround CAM * normalized CAM * Original Image. does norm -> multiply order matter?
        # background_mask = singel_cam_masks[-1] # Background CAM
        background_mask = mask[batch_idx].clone()
        background_mask[background_mask == 0] = 0.2
        soft_apply = sum_masks * background_mask # cam predicts * background constraints
        
        num_channels = image.shape[1]
        for s in range(0, num_channels, 3):
            inputs_after_soft_addon = soft_apply * image[batch_idx, s:s+3,] # [3, w, h]
            # normilize the input image after addon soft map on origin input (both origin & gan)
            soft_min, soft_max = inputs_after_soft_addon.min(), inputs_after_soft_addon.max()
            inputs_after_soft_addon.add_(-soft_min).div_(soft_max - soft_min + 1e-5)
            updated_input_tensor[batch_idx, s:s+3,] = inputs_after_soft_addon
    return updated_input_tensor

# Not use in current version
def refine_input_by_background_cam(args, model, image, mask, cam, aug_smooth=True):
    outputs = model(image)
    bacth_bg_preds = (outputs[:, -1] > 0.5) * 1 # [batch] -> (0,1)
    bg_cls = len(OrgLabels) - 1
    targets = [ClassifierOutputTarget(bg_cls)] * len(image) # for all in batch return the background cam
    batch_grayscale_cam = cam(input_tensor=image,targets=targets,eigen_smooth=False, aug_smooth=aug_smooth)
    grayscale_tensor = torch.from_numpy(batch_grayscale_cam).to(args.device)
    grayscale_tensor = grayscale_tensor.unsqueeze(1).repeat(1, 3, 1, 1) # extend gray to 3 channels [batch, 3, w, h]
    
    updated_input_tensor = image.clone() # [batch, c, w, h]
    '''Apply background cam on original image'''
    # BackGround CAM * Original Image for predicted 1 background image only.
    for batch_idx, bg_pred in enumerate(bacth_bg_preds):
        soft_apply = grayscale_tensor[batch_idx] * image[batch_idx, :3,] if bg_pred > 0 else image[batch_idx, :3] # [3, w, h]
        soft_min, soft_max = soft_apply.min(), soft_apply.max()
        soft_apply.add_(-soft_min).div_(soft_max - soft_min + 1e-5)
        updated_input_tensor[batch_idx, :3,] = soft_apply
    return updated_input_tensor

def get_pseudo_label(params, cam):
    inputs, batch_preds, updated_image = params['inputs'], params['batch_preds'], params['refined']
    pseudo_labels = []
    for i, pred in enumerate(batch_preds):
        orig_image = inputs["image"][i][0].clone()
        orig_mask = inputs['mask'][i][0].clone()
        w, h = inputs["image"][i].shape[-2], inputs["image"][i].shape[-1]
        pred_classes = [i for i,v in enumerate(pred) if v > 0.5]
        all_grey = [np.zeros((w, h))] * (len(OrgLabels) + 1)
        for cls in pred_classes:
            targets = [ClassifierOutputTarget(cls)]
            grayscale_cam = cam(input_tensor=updated_image,targets=targets,eigen_smooth=False, aug_smooth=True)
            grayscale_cam = grayscale_cam[0, :]

            grey_thred = grayscale_cam.copy()
            grey_thred[grey_thred < 0.3] = 0
            
            strict_grey = grey_thred.copy()
            if OrgLabels[cls] in ['IRF', 'SRF']:
                strict_grey[(orig_image > 0.2) | (orig_mask == 0)] = 0 #keep bubble region (dark)
            else: strict_grey[(orig_image < 0.5) | (orig_mask == 0)] = 0 # hrd, ez keep the light layer region
            if OrgLabels[cls] != 'BackGround': # dont illustrate the background
                all_grey[cls + 1] = strict_grey
        
        labels = np.argmax(np.array(all_grey), axis=0) # (256, 256) with 0-6 labels
        pseudo_labels.append(labels)
    return torch.LongTensor(pseudo_labels)
    

if __name__ == "__main__":
    backbone = models.resnet18(pretrained=True)
    num_class = len(OrgLabels)
    device="cuda" if torch.cuda.is_available() else "cpu"
