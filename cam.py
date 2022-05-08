from operator import mod
from statistics import mode
from turtle import pd
from model import MultiTaskModel
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
import os, time
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from oct_utils import OrgLabels, type_color
from dataset import OCTDataset, valid_transform
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from torch.autograd import Variable
import torchvision.utils as vutils
from torch import nn


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def save_cam_during_train(params, cam):
    args, epoch, fold, inputs, batch_preds, updated_image = params['args'], params['epoch'], params['fold'], params['inputs'], params['batch_preds'], params['refined']
    for i, pred in enumerate(batch_preds):
        rgb_img = (np.float32(inputs["image"][i][:3].permute(1, 2, 0)))
        img_path = inputs["path"][i].split('/')[-1]
        save_path = os.path.join(args.save_folder, f'fold-{fold}', 'iteration', '{}'.format(img_path.split('.')[0]))
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
            input_tensor  = inputs["image"].to(args.device)
            grayscale_cam = cam(input_tensor=input_tensor,targets=targets,eigen_smooth=False, aug_smooth=True)
            grayscale_cam = grayscale_cam[0, :]

            grey_thred = grayscale_cam.copy()
            # test = grey_thred * rgb_img[...,0]
            # cv2.imwrite(save_path + '/test{}.jpg'.format(cls), (test * 255).astype(np.uint8))
            grey_thred[grey_thred < 0.8] = 0
            if OrgLabels[-1] == 'BackGround' and cls != (len(OrgLabels)-1): # dont illustrate the background
                all_grey[cls + 1] = grey_thred
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
    print(batch_preds)
    # softmax = nn.Softmax()
    for i, pred in enumerate(batch_preds):
        rgb_img = (np.float32(inputs["image"][i][:3].permute(1, 2, 0)))
        img_path = inputs["path"][i].split('/')[-1]
        save_path = os.path.join(args.save_inference, '{}'.format(img_path.split('.')[0]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
        true_classes = [i for i,v in enumerate(inputs["labels"][i]) if v > 0.5]
        truth_label = [OrgLabels[cls] for cls in true_classes]
        truth_label = '_'.join(truth_label)
        w, h = inputs["image"][i].shape[-2], inputs["image"][i].shape[-1]
        save_img = inputs["image"][i].reshape(-1,3,w, h)
        vutils.save_image(save_img, save_path + '/orig_{}.jpg'.format(truth_label), normalize=True)

        pred_classes = [i for i,v in enumerate(pred) if v > 0.5]
        print(pred_classes)
        save_cam_in_row = []
        save_class_name = ''
        
        all_grey = [np.zeros((w, h))] * (len(OrgLabels) + 1)
        if 'BackGround' in OrgLabels:
            idx_bg = len(OrgLabels) - 1
            targets = [ClassifierOutputTarget(idx_bg)]
            input_tensor  = inputs["image"].to(args.device)
            grayscale_cam = cam(input_tensor=input_tensor,targets=targets,eigen_smooth=False, aug_smooth=True)
            grayscale_cam_bg = grayscale_cam[0, :]
            grayscale_cam_bg[grayscale_cam_bg < 0.2] = 0
            
        for cls in pred_classes:
            targets = [ClassifierOutputTarget(cls)]
            input_tensor  = inputs["image"].to(args.device)
            grayscale_cam = cam(input_tensor=input_tensor,targets=targets,eigen_smooth=False, aug_smooth=True)
            grayscale_cam = grayscale_cam[0, :]
            
            # remove the heat out of background prediction
            grayscale_cam[grayscale_cam_bg==0] = 0

            grey_thred = grayscale_cam.copy()
            grey_thred[grey_thred < 0.8] = 0
            
            if OrgLabels[-1] == 'BackGround' and cls != (len(OrgLabels)-1):
                all_grey[cls + 1] = grey_thred
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
        import pdb; pdb.set_trace()
               
def save_cam_results(params, is_inference=False):
    args = params['args']
    model = params['model']
    # batch_preds [BC] B: batch, C: Class
    if args.cam_type == 'gradcam':
        target_layers = [model.base_model.layer4[-1]]
        cam = GradCAM(model=model, use_cuda=args.device, target_layers=target_layers)
    else: 
        target_layers = [model.base_model.layer4[-1]]
        cam = GradCAM(model=model, use_cuda=args.device, target_layers=target_layers)
    if is_inference:
        save_cam_for_inference(params, cam)
        return
    save_cam_during_train(params, cam)
    
if __name__ == "__main__":
    backbone = models.resnet18(pretrained=True)
    num_class = len(OrgLabels)
    device="cuda" if torch.cuda.is_available() else "cpu"
    # model= MultiTaskModel(backbone, num_class).to(device)

    # # if device == "cuda":
    # #     print("GPU: ", torch.cuda.device_count())
    # #     model = torch.nn.DataParallel(model, device_ids=list(
    # #         range(torch.cuda.device_count()))).cuda()
        
    # checkpoint = torch.load('outputs/results_5labels/fold-0/50.pwf')    
    # model.load_state_dict(checkpoint['state_dict'])

    # model.eval()
    # target_layers = [model.base_model.layer4[-1]]

    # cam = GradCAM(model=model, use_cuda=device, target_layers=target_layers)
    # cam_extractor = SmoothGradCAMpp(model)

    # Number = 1

    # root_dirs = ["examples/data"]
    # dataset = OCTDataset(root_dirs, transform=valid_transform())
    # input_tensor = dataset[Number]["image"]
    # rgb_img = (np.float32(input_tensor.permute(1, 2, 0)))
    # print(dataset[Number]["labels"], dataset[Number]["path"])
    # plt.imshow(rgb_img); plt.axis('off'); plt.tight_layout(); plt.show()
    # plt.savefig(f'examples/temp/ori.jpg' )
    # input_tensor = input_tensor.unsqueeze(0).to(device)
    # output = model(input_tensor)
    # print(output)
    

    # # ['srf', 'irf', 'ezAtt', 'ezDis', 'hrd', 'rpe', 'rt', 'dril']
    # # activation_map = cam_extractor(output.squeeze(0).argmax().item(), output)
    # targets = [ClassifierOutputTarget(7)]
    # print(input_tensor.shape, targets)
    # grayscale_cam = cam(input_tensor=input_tensor,targets=targets,eigen_smooth=False, aug_smooth=True)
    # grayscale_cam = grayscale_cam[0, :]
    # # grayscale_cam[grayscale_cam<0.5] =0 

    # # Visualize the raw CAM
    # # plt.imshow(grayscale_cam); plt.axis('off'); plt.tight_layout(); plt.show()
    # # print(grayscale_cam, grayscale_cam.shape, rgb_img.shape)
    # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # # plt.imshow(visualization); plt.axis('off'); plt.tight_layout(); plt.show()
    # # plt.savefig('test.png')

    # cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=device)
    # gb = gb_model(input_tensor, target_category=None)
    # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)

    # save_root = 'examples/temp'
    # cv2.imwrite(f'%s/cam.jpg' % save_root, cam_image)
    # cv2.imwrite(f'%s/gb.jpg' % save_root, gb)
    # cv2.imwrite(f'%s/cam_gb.jpg'% save_root, cam_gb)
