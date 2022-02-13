import torch
import torch.nn as nn
from torch.utils import data
from torchvision.models import vgg19
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

# use the ImageNet transformation
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# define a 1 image dataset
dataset = datasets.ImageFolder(root='/./dataset_DME/3/images/DME-2351012-6/', transform=transform)

# define the dataloader to load that single image
# dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)

# import os
# print(os.chdir('datase_DME'))