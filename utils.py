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

'''  0.6 has the highest mIoU from SEAM results, but our cam is too shallow, we use 0.3'''
out_cam_pred_alpha = 0.35

def get_num_classes():
    if 'BackGround' in OrgLabels:
        return len(OrgLabels) - 1
    return len(OrgLabels)

def convert_resc_labels(img):
    # 0 background, 
    # 1 lesion(need turn to background),  1 -> 0
    # 0.74 SRF(need turn to 1),           0.74 -> 1
    # 0.51 PED(need turn to 2)            0.51 -> 2
    img[img == 1] = 0
    img[img > 0.7] = 1
    img[(img < 0.7) & (img > 0.4)] = 2
    return img.numpy()

def post_process_cam(cls, grayscale_cam, orig_mask, orig_image):
    grey_thred = grayscale_cam.copy()
    mask_clone = orig_mask.clone() # NEUROSENSORY RETINA only (ILM to RPE)
    mask_clone[mask_clone == 0] = 0.5
    grey_thred = grey_thred * mask_clone.cpu().numpy()
    grey_thred[grey_thred < 0.2] = 0  # (h, w)

    # if OrgLabels[cls] in ['IRF', 'SRF', 'PED']:
    #     grey_thred[(orig_image > 0.2)] = 0 #keep bubble region (dark)
    # else: grey_thred[(orig_image < 0.5)] = 0 # hrd, ez keep the light layer region
                
    # if OrgLabels[cls] in ['IRF', 'SRF', 'PED']:
    #     grey_thred[orig_image > 0.15] = 0 #keep bubble region (dark) #| (orig_mask == 0)
    # elif OrgLabels[cls] == 'EZ': grey_thred[orig_image < 0.5] = 0 # hrd, ez keep the light layer region
    return grey_thred

def get_cam_results_per_class(cam, inputs, updated_image, pred, i):
    rgb_img = (np.float32(inputs["image"][i][:3].permute(1, 2, 0)))
    orig_image = inputs['image'][i][0]
    orig_mask = inputs['mask'][i][0].clone()
    save_class_name = ''
    save_cam_in_row = []
    w, h = orig_image.shape[-2], orig_image.shape[-1]
    bg_score = [np.ones_like(orig_image) * out_cam_pred_alpha] * (get_num_classes() + 1) # resc = 1 + 2 length
    pred_classes = [i for i,v in enumerate(pred) if v > 0.5]
    for cls in pred_classes:
        if OrgLabels[cls] == 'BackGround':
            continue
        targets = [ClassifierOutputTarget(cls)]
        grayscale_cam = cam(input_tensor=updated_image,targets=targets,eigen_smooth=False, aug_smooth=False) #(h,w)
        grayscale_cam = grayscale_cam[0, :]
        bg_score[cls + 1] = post_process_cam(cls, grayscale_cam, orig_mask, orig_image)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        save_cam_in_row.append(cam_image)
        save_class_name =  save_class_name + '_' + OrgLabels[cls]
        # import pdb; pdb.set_trace()
    labels = np.argmax(np.array(bg_score), axis=0) # [0 - num_class]
    
    color_mask = np.zeros((w, h, 3))
    for i in range(1, get_num_classes() + 1):
        mask = labels == i
        color_mask[:,:,][mask] = type_color[i]
    color_mask = cv2.cvtColor(color_mask.astype(np.uint8), cv2.COLOR_BGR2RGB)
    save_cam_in_row.append(color_mask)
    dst = cv2.addWeighted((rgb_img * 255).astype(np.uint8), 1, color_mask.astype(np.uint8), 0.7, 0)
    save_cam_in_row.append(dst)
    
    return save_class_name, save_cam_in_row, labels

def save_cam_during_train(params, cam):
    args, epoch, inputs, batch_preds, updated_image = params['args'], params['epoch'], params['inputs'], params['batch_preds'], params['refined']
    non_background_names = [x for x in OrgLabels if 'BackGround' != x]
    lesion_classes = [OrgLabels.index(name) for name in non_background_names]
    for i, pred in enumerate(batch_preds):
        ground_true_classes = [i for i,v in enumerate(inputs['labels'][i]) if v > 0.5]
        # only calculate and save for ground truth lesion images
        if not len(set(lesion_classes) & set(ground_true_classes)):
            continue        
        img_path = inputs["path"][i].split('/')[-1]
        save_path = os.path.join(args.save_folder, 'iteration', '{}'.format(img_path.split('.')[0]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
        true_classes = [i for i,v in enumerate(inputs["labels"][i]) if v > 0.5]
        truth_label = [OrgLabels[cls] for cls in true_classes]
        truth_label = '_'.join(truth_label)
        w, h = inputs["image"][i].shape[-2], inputs["image"][i].shape[-1]
        save_img = inputs["image"][i].reshape(-1,3,w, h)
        orig_mask = inputs['mask'][i].clone().reshape(-1,3,w, h)
        save_img = torch.cat([save_img, orig_mask], 0)
        if 'annot' in inputs:
            # import pdb; pdb.set_trace()
            save_img = torch.cat([save_img, inputs["annot"][i].reshape(-1,3,w, h)], 0)
        save_updated_img = updated_image[i].reshape(-1,3,w, h)
        vutils.save_image(save_img, save_path + '/orig_{}.jpg'.format(truth_label), normalize=True, scale_each=True) # scale_each limit normalize for each independently in batch
        vutils.save_image(save_updated_img, save_path + '/epoch{0}_refined_{1}.jpg'.format(epoch, truth_label), normalize=True, scale_each=True)
        save_class_name, save_cam_in_row, _ = get_cam_results_per_class(cam, inputs, updated_image, pred, i)
        
        if (len(save_cam_in_row)):
            im_h = cv2.hconcat(save_cam_in_row)
            cv2.imwrite(save_path + '/epoch{0}_{1}.jpg'.format(epoch, save_class_name), im_h)

def save_cam_for_inference(params, cam):
    args, inputs, batch_preds, updated_image, classify_model = params['args'], params['inputs'], params['batch_preds'], params['refined'], params['model']
    non_background_names = [x for x in OrgLabels if 'BackGround' != x]
    lesion_classes = [OrgLabels.index(name) for name in non_background_names]
    ready_pred_4d = []
    gt = []
    for i, pred in enumerate(batch_preds):
        ground_true_classes = [i for i,v in enumerate(inputs['labels'][i]) if v > 0.5]
        # only calculate and save for ground truth lesion images
        if not len(set(lesion_classes) & set(ground_true_classes)):
            continue
        img_path = inputs["path"][i].split('/')[-1]
        save_path = os.path.join(args.save_inference, '{}'.format(img_path.split('.')[0]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
        true_classes = [i for i,v in enumerate(inputs["labels"][i]) if v > 0.5]
        truth_label = [OrgLabels[cls] for cls in true_classes]
        truth_label = '_'.join(truth_label)
        w, h = inputs["image"][i].shape[-2], inputs["image"][i].shape[-1]
        
        save_img = torch.cat([inputs["image"][i].reshape(-1,3,w, h).to(args.device), updated_image[i].reshape(-1,3,w, h)], 0)
        orig_mask = inputs['mask'][i].clone().reshape(-1,3,w, h).to(args.device)
        save_img = torch.cat([save_img, orig_mask], 0)

        if 'annot' in inputs:
            save_img = torch.cat([save_img, inputs["annot"][i].reshape(-1,3,w, h).to(args.device)], 0)
        vutils.save_image(save_img, save_path + '/orig_{}.jpg'.format(truth_label), normalize=True, scale_each=True)

        save_class_name, save_cam_in_row, labels = get_cam_results_per_class(cam, inputs, updated_image, pred, i)
        
        # TODO: need remove this if check, this is only for skip the cam disappear images
        if save_cam_in_row[-2].max() == 0:
            continue
        gt.append(convert_resc_labels(inputs["annot"][i,0].clone()))
        ready_pred_4d.append(labels)
        if (len(save_cam_in_row)):
            im_h = cv2.hconcat(save_cam_in_row)
            cv2.imwrite(save_path + '/{0}.jpg'.format(save_class_name), im_h)
        # import pdb; pdb.set_trace()
    return gt, ready_pred_4d
  
def save_cam_results(params, is_inference=False):
    cam = params['cam']
    # target_layers = [params['model'].multi_task_model.base_model[-1][-3]]
    # print(target_layers)
    # cam = GradCAM(model=params['model'], use_cuda=params['args'].device, target_layers=target_layers)
    # batch_preds [BC] B: batch, C: Class
    if is_inference:
        return save_cam_for_inference(params, cam)
    save_cam_during_train(params, cam)


def save_models(args, epoch, cam_model, cam_optimizer, is_best=False):
    save_path = f'./{args.save_folder}/weights'
    save_name = 'best' if is_best else epoch + 1
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
        'epoch': epoch,
        'args': args,
        'state_dict': cam_model.state_dict(),
        'optimizer': cam_optimizer.state_dict(),
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