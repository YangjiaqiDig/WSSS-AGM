from operator import mod
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
from oct_utils import OrgLabels
from dataset import OCTDataset, valid_transform
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def save_cam_results(args, epoch, model, inputs, batch_preds):
    # batch_preds [BC] B: batch, C: Class
    if args.cam_type == 'gradcam':
        target_layers = [model.base_model.layer4[-1]]
        cam = GradCAM(model=model, use_cuda=args.device, target_layers=target_layers)
    else: 
        target_layers = [model.base_model.layer4[-1]]
        cam = GradCAM(model=model, use_cuda=args.device, target_layers=target_layers)
    for i, pred in enumerate(batch_preds):
        pred_classes = [i for i,v in enumerate(pred) if v > 0.5]
        img_path = pred["path"]
        for cls in pred_classes:
            targets = [ClassifierOutputTarget(cls)]
            grayscale_cam = cam(input_tensor=input_tensor,targets=targets,eigen_smooth=False, aug_smooth=True)
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam[grayscale_cam<0.5] =0 
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'cam.jpg', cam_image)
    
    
if __name__ == "__main__":
    backbone = models.resnet18(pretrained=True)
    num_class = len(OrgLabels)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model= MultiTaskModel(backbone, num_class).to(device)

    # if device == "cuda":
    #     print("GPU: ", torch.cuda.device_count())
    #     model = torch.nn.DataParallel(model, device_ids=list(
    #         range(torch.cuda.device_count()))).cuda()
        
    checkpoint = torch.load('output/results_dr/fold-0/50.pwf')
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    target_layers = [model.base_model.layer4[-1]]


    cam = GradCAM(model=model, use_cuda=device, target_layers=target_layers)
    cam_extractor = SmoothGradCAMpp(model)

    Number = 1

    root_dirs = ["our_dataset/dataset_DR"]
    dataset = OCTDataset(root_dirs, transform=valid_transform())
    input_tensor = dataset[Number]["image"]
    rgb_img = (np.float32(input_tensor.permute(1, 2, 0)))
    print(dataset[Number]["labels"], dataset[Number]["path"])
    plt.imshow(rgb_img); plt.axis('off'); plt.tight_layout(); plt.show()

    input_tensor = input_tensor.unsqueeze(0).to(device)
    output = model(input_tensor)
    print(output)

    # ['srf', 'irf', 'ezAtt', 'ezDis', 'hrd', 'rpe', 'rt', 'dril']
    # activation_map = cam_extractor(output.squeeze(0).argmax().item(), output)
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=input_tensor,targets=targets,eigen_smooth=False, aug_smooth=True)
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam[grayscale_cam<0.5] =0 

    # Visualize the raw CAM
    plt.imshow(grayscale_cam); plt.axis('off'); plt.tight_layout(); plt.show()
    # print(grayscale_cam, grayscale_cam.shape, rgb_img.shape)
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    plt.imshow(visualization); plt.axis('off'); plt.tight_layout(); plt.show()
    # plt.savefig('test.png')

    cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=device)
    gb = gb_model(input_tensor, target_category=None)
    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    save_root = 'examples/temp'
    cv2.imwrite(f'%s/cam.jpg' % save_root, cam_image)
    cv2.imwrite(f'%s/gb.jpg' % save_root, gb)
    cv2.imwrite(f'%s/cam_gb.jpg'% save_root, cam_gb)