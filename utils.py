import os
import torch
import numpy as np
from options import Configs
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, GuidedBackpropReLUModel
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
import torchvision.models as models
import torchvision.utils as vutils
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F
from PIL import Image

type_color = {
    0: [0, 0, 0], # black
    1: [255, 0, 0], # srf red
    2: [0, 255, 0], # irf green
    3: [0, 0, 255], # ez blue
    4: [255, 255, 0], # hrd yellow
    5: [255, 165, 0], # rpe orange
    6: [255, 0, 255] # back ground pink
}
OrgLabels = Configs().get_labels()#['SRF', 'IRF', 'EZ', 'HRD',  'RPE', 'BackGround']

'''  0.6 has the highest mIoU from SEAM results'''
out_cam_pred_alpha = 0.7

def get_num_classes():
    if 'BackGround' in OrgLabels:
        return len(OrgLabels) - 1
    return len(OrgLabels)

def normalized_batch_tensor(t):
    orig_size = t.shape
    t = t.view(orig_size[0], -1)
    t -= t.min(1, keepdim=True)[0]
    t /= t.max(1, keepdim=True)[0]
    t = t.view(orig_size)
    return t

def diff_map_for_att(orig_tensor, gan_tensor, mask_tensor):
    # batch, channel, h, w
    normalized_orig = orig_tensor.clone()
    normalized_gan = gan_tensor.clone()
    
    normalized_orig = normalized_batch_tensor(normalized_orig)
    normalized_gan = normalized_batch_tensor(normalized_gan)
    
    abs_diff = torch.abs(normalized_orig - normalized_gan)
    mask_out_diff = abs_diff * mask_tensor

    return mask_out_diff

# input is array H * W
def convert_resc_pixel2image(label_pixel_arr):
    # back: 0, ped: 128, srf: 191, retinal: 255
    image_label = {'SRF': 0, 'PED': 0, 'lesion': 0, 'health': 0, 'BackGround': 1}
    labels = np.unique(label_pixel_arr)
    if len(labels) == 1:
        image_label['health'] += 1
    if 128 in labels:
        image_label['PED'] += 1
    if 191 in labels:
        image_label['SRF'] += 1
    if 255 in labels:
        image_label['lesion'] +=1 
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

def post_process_cam(resized_cam, orig_mask):
    mask_clone = orig_mask.copy() # NEUROSENSORY RETINA only (ILM to RPE)
    mask_clone[mask_clone == 0] = 0.5
    masked_cam = resized_cam * mask_clone
    
    cam_max = masked_cam.max()
    cam_min = masked_cam.min()
    norm_masked_cam = (masked_cam - cam_min - 1e-5) / (cam_max - cam_min + 1e-5)
    
    return norm_masked_cam
class CAMGeneratorAndSave():
    def __init__(self, opts, multi_task_model, epoch=None) -> None:
        self.opts = opts
        self.multi_task_model = multi_task_model
        self.epoch = epoch
        non_background_names = [x for x in OrgLabels if 'BackGround' != x]
        self.lesion_classes = [OrgLabels.index(name) for name in non_background_names]
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def get_cam_results_per_class(self, orig_img, orig_mask, ground_true_classes):
        save_cam_in_row = []
        rgb_img = (orig_img / 255).copy()
        if (rgb_img.ndim == 2):
            rgb_img = np.repeat(rgb_img[..., np.newaxis], 3, -1)
            
        target_layers = [self.multi_task_model.module.ClassNet.postatt]#[self.multi_task_model.module.base_model[-1][-1]] # .module. if use dataparallel
        norm_cam_list = []
        for cls in range(len(OrgLabels)):
            '''Generate all cams except the ones that are not in ground true classes'''    
            if OrgLabels[cls] == 'BackGround' or cls not in ground_true_classes:
                continue
            targets = [ClassifierOutputTarget(cls)]
            
            with GradCAM(model=self.multi_task_model, use_cuda="cuda", target_layers=target_layers) as cam:
                cam.model.module.assign_conditions(True, False)
                grayscale_cam = cam(input_tensor=self.updated_image,targets=targets, eigen_smooth=False, aug_smooth=False) #(h,w) and  [0, x]
            # import pdb; pdb.set_trace()
            cam_res = grayscale_cam[0, :]
            resized_cam = F.interpolate(torch.from_numpy(cam_res).unsqueeze(0).unsqueeze(0), size=orig_mask.shape, mode='bilinear', align_corners=False)[0,0].numpy()
            single_norm_cam = post_process_cam(resized_cam, orig_mask)
            norm_cam_list.append(single_norm_cam)

            visualization = show_cam_on_image(rgb_img, resized_cam, use_rgb=True)
            cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            save_cam_in_row.append(cam_image)
        pred_with_bg_score = [np.ones_like(norm_cam_list[0]) * out_cam_pred_alpha] + norm_cam_list # (K + 1, [h, w])
        pred_labels = np.argmax(np.array(pred_with_bg_score), axis=0) # [0 - num_class]
        
        color_mask = np.zeros_like(rgb_img)
        for i_cls in range(1, get_num_classes() + 1):
            mask = pred_labels == i_cls
            color_mask[:,:,][mask] = type_color[i_cls]
        color_mask = cv2.cvtColor(color_mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
        save_cam_in_row.append(color_mask)
        dst = cv2.addWeighted((rgb_img * 255).astype(np.uint8), 1, color_mask.astype(np.uint8), 0.7, 0)
        save_cam_in_row.append(dst)
        
        return save_cam_in_row, pred_labels
    
    def save_cam_process(self, ground_true_classes, batch_nb):
        img_name = self.inputs["path"][batch_nb].split('/')[-1]
        # 0-255 pixel value, numpy
        orig_img = np.asarray(Image.open(self.inputs["path"][batch_nb]))
        mask_path = os.path.join(self.opts.mask_dir, 'valid', img_name)
        orig_mask = np.asarray(Image.open(mask_path))[...,0]
        annot_path = os.path.join(self.opts.root_dirs, 'valid/label_images', img_name)
        orig_annot = np.asarray(Image.open(annot_path))
        
        true_classes = [i for i,v in enumerate(self.inputs["labels"][batch_nb]) if v > 0.5]
        truth_label = [OrgLabels[cls] for cls in true_classes]
        truth_label = '_'.join(truth_label)
        save_img = [orig_img, orig_mask, orig_annot]
        save_image_h = cv2.hconcat(save_img)
        
        if self.is_inference:
            save_path = os.path.join(self.opts.save_inference, '{}'.format(img_name.split('.')[0]))
        else: save_path = os.path.join(self.opts.save_folder, 'iteration', '{}'.format(img_name.split('.')[0]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
        cv2.imwrite(save_path + '/orig_{}.jpg'.format(truth_label), save_image_h)
        
        if not self.is_inference:
            w, h = self.opts.is_size[0], self.opts.is_size[1]
            save_updated_img = self.updated_image[batch_nb].reshape(-1,3,w, h)
            vutils.save_image(save_updated_img, save_path + '/epoch{0}_refined_{1}.jpg'.format(self.epoch, truth_label), normalize=True, scale_each=True)
        
        save_cam_in_row, pred_labels = self.get_cam_results_per_class(orig_img, orig_mask, ground_true_classes)
        
        return convert_resc_labels(orig_annot), pred_labels, save_cam_in_row, save_path
    
    def get_cam_and_save(self, params, is_inference=False):
        # batch_preds [BC] B: batch, C: Class
        self.inputs, self.updated_image, batch_preds = params['inputs'], params['refined'], params['batch_preds']
        ready_pred_4d = []
        gt = []
        self.is_inference = is_inference
        for batch_nb, single_pred in enumerate(batch_preds):
            ground_true_classes = [i for i,v in enumerate(self.inputs['labels'][batch_nb]) if v > 0.5]
            # only calculate and save for ground truth lesion images
            if not len(set(self.lesion_classes) & set(ground_true_classes)):
                continue
            gt_labels, pred_labels, save_cam_in_row, save_path = self.save_cam_process(ground_true_classes, batch_nb)
            gt.append(gt_labels)
            ready_pred_4d.append(pred_labels)
            
            single_pred_cls = [i for i,v in enumerate(single_pred) if v > 0.5]
            save_class_name =  '_'.join([OrgLabels[cls] for cls in single_pred_cls if OrgLabels[cls] != 'BackGround'])
            if (len(save_cam_in_row)):
                im_h = cv2.hconcat(save_cam_in_row)
                if self.is_inference:
                    cv2.imwrite(save_path + '/pred_{0}.jpg'.format(save_class_name), im_h)
                else: cv2.imwrite(save_path + '/epoch{0}_{1}.jpg'.format(self.epoch, save_class_name), im_h)
        return gt, ready_pred_4d


def save_models(args, epoch, multi_task_model, multi_optimizer, is_best=False):
    save_path = f'./{args.save_folder}/weights'
    save_name = 'best' if is_best else epoch + 1
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
        'epoch': epoch,
        'args': args,
        'state_dict': multi_task_model.module.state_dict(),
        'optimizer': multi_optimizer.state_dict(),
    }, save_path + "/{0}.pwf".format(save_name)) 

def save_tensorboard(tb, loss_dict, mark_epoch, include_valid):
    tb.add_scalar('Loss/Train', loss_dict['total_train_loss'], mark_epoch+1)
    tb.add_scalar("ROC/Train",  loss_dict['total_train_roc'], mark_epoch+1)
    for acc_type in ['acc', 'f1m']:
        tb.add_scalar("Train Accuracy/{}".format(acc_type), loss_dict['total_train_acc_matrix'][acc_type], mark_epoch+1)
    for label_type in OrgLabels:
        tb.add_scalar("Train Class Acc/{}".format(label_type), loss_dict['total_train_acc_matrix'][label_type], mark_epoch+1)
        tb.add_scalar("Train Class ROC/{}".format(label_type), loss_dict['total_train_roc_matrix'][label_type], mark_epoch+1)
    if include_valid:
        tb.add_scalar('Loss/Valid', loss_dict['total_val_loss'], mark_epoch+1)
        tb.add_scalar("ROC/Valid",  loss_dict['total_val_roc'], mark_epoch+1)
        for acc_type in ['acc', 'f1m']:
            tb.add_scalar("Val Accuracy/{}".format(acc_type),  loss_dict['total_val_acc_matrix'][acc_type], mark_epoch+1)
        for label_type in OrgLabels:
            tb.add_scalar("Val Class Acc/{}".format(label_type), loss_dict['total_val_acc_matrix'][label_type], mark_epoch+1)
            tb.add_scalar("Val Class ROC/{}".format(label_type), loss_dict['total_val_roc_matrix'][label_type], mark_epoch+1)

    tb.close()

if __name__ == "__main__":
    backbone = models.resnet18(pretrained=True)
    num_class = len(OrgLabels)
    device="cuda" if torch.cuda.is_available() else "cpu"

    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    model = models.resnet50(pretrained=True)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [model.layer4]

    # rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    # rgb_img = np.float32(rgb_img) / 255
    # input_tensor = preprocess_image(rgb_img,
    #                                 mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])


    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category (for every member in the batch) will be used.
    # You can target specific categories by
    # targets = [e.g ClassifierOutputTarget(281)]
    targets = None

    # Using the with statement ensures the context is freed, and you can
    # recreate different CAM objects in a loop.
    
    # cam_algorithm = methods[args.method]
    # with cam_algorithm(model=model,
    #                    target_layers=target_layers,
    #                    use_cuda=args.use_cuda) as cam:

    #     # AblationCAM and ScoreCAM have batched implementations.
    #     # You can override the internal batch size for faster computation.
    #     cam.batch_size = 32
    #     grayscale_cam = cam(input_tensor=input_tensor,
    #                         targets=targets,
    #                         aug_smooth=args.aug_smooth,
    #                         eigen_smooth=args.eigen_smooth)

    #     # Here grayscale_cam has only one image in the batch
    #     grayscale_cam = grayscale_cam[0, :]

    #     cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    #     # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
    #     cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # gb = gb_model(input_tensor, target_category=None)

    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)

    # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
    # cv2.imwrite(f'{args.method}_gb.jpg', gb)
    # cv2.imwrite(f'{args.method}_cam_gb.jpg', cam_gb)