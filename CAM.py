from operator import mod
from model import MultiTaskModel
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import torch
import os, time
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


os.environ['CUDA_VISIBLE_DEVICES'] = '3'

device="cuda" if torch.cuda.is_available() else "cpu"
model= MultiTaskModel().to(device)

# if device == "cuda":
#     print("GPU: ", torch.cuda.device_count())
#     model = torch.nn.DataParallel(model, device_ids=list(
#         range(torch.cuda.device_count()))).cuda()
    
checkpoint = torch.load('results_dr/fold-0/50.pwf')
model.load_state_dict(checkpoint['state_dict'])

# print(model)
model.train()
target_layers = [model.base_model.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device)
print(ClassifierOutputTarget)
# targets = [ClassifierOutputTarget(281)]
# print(targets)
file_list = glob.glob("%s/*.npy" % 'train_dr')
data_path = file_list[0]
print(data_path)
data_pack = np.load(data_path, allow_pickle=True)
dict_data = data_pack.item()
rgb_img = dict_data["image"]

transform_seq = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((500,500)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
input_tensor = transform_seq(rgb_img)
input_tensor = torch.unsqueeze(input_tensor, axis=0).to(device)
print(input_tensor.is_cuda )

    
grayscale_cam = cam(input_tensor=input_tensor)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)