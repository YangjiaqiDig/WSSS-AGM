import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

temp_model = torchvision.models.resnet18()
print(temp_model)