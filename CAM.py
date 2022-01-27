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
from oct_utils import LABELS
from dataset import OCTDataset, valid_transform
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
backbone = models.resnet18(pretrained=True)
num_class = len(LABELS)
device="cuda" if torch.cuda.is_available() else "cpu"
model= MultiTaskModel(backbone, num_class).to(device)

# if device == "cuda":
#     print("GPU: ", torch.cuda.device_count())
#     model = torch.nn.DataParallel(model, device_ids=list(
#         range(torch.cuda.device_count()))).cuda()
    
checkpoint = torch.load('results_dr/fold-0/50.pwf')
model.load_state_dict(checkpoint['state_dict'])

model.eval()
target_layers = [model.base_model.layer4[-1]]


cam = GradCAM(model=model, use_cuda=device, target_layers=target_layers)
cam_extractor = SmoothGradCAMpp(model)

# targets = [ClassifierOutputTarget(281)]
# print(targets)
root_dir = "cam_test"
dataset = OCTDataset(root_dir, transform=valid_transform())
input_tensor = dataset[1]["image"]
rgb_img = (np.float32(input_tensor.permute(1, 2, 0)))
print(dataset[1]["labels"])
plt.imshow(rgb_img); plt.axis('off'); plt.tight_layout(); plt.show()

input_tensor = input_tensor.unsqueeze(0).to(device)
output = model(input_tensor)
print(output)
# ['srf', 'irf', 'ezAtt', 'ezDis', 'hrd', 'rpe', 'rt', 'dril']
# activation_map = cam_extractor(output.squeeze(0).argmax().item(), output)
target = [ClassifierOutputTarget(7)]
grayscale_cam = cam(input_tensor=input_tensor,targets=target,eigen_smooth=False, aug_smooth=True)
grayscale_cam = grayscale_cam[0, :]
# Visualize the raw CAM
plt.imshow(grayscale_cam); plt.axis('off'); plt.tight_layout(); plt.show()
# plt.savefig('cam_test/irf.png')
# print(grayscale_cam, grayscale_cam.shape, rgb_img.shape)
# sss
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
plt.imshow(visualization); plt.axis('off'); plt.tight_layout(); plt.show()
plt.savefig('cam_test/9_dril.png')
