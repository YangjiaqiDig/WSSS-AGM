import torch
import torch.nn as nn
# from torch.utils import data
# from torchvision.models import vgg19
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
# from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from dataset import OCTDataset
from tqdm.notebook import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

######## PREPARING THE IMAGES

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

dataset = OCTDataset(["dataset_DR", "dataset_DME/1", "dataset_DME/3"], transform=image_transform)
train_dataset, test_dataset = random_split(dataset, (1680, 426))

train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=1)
val_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

# print("The sample size of train dataset: ",len(train_dataset))
# print("The sample size of test dataset: ",len(test_dataset))

for x in train_loader:
    break


# print(x['image'][0].shape)
# print(x['labels'][0])


# everytime for image --> x[i]['image'] and for label --> x[i]['labels][0]
# print("Size of an image: ", x["image"].shape)
# print("Label of the first image: ", x['labels'][0])
# print("Label tensor shape is: ", x['labels'][0].shape)

######## PLOT ONE IMAGE

# plt.imshow(x["image"].view(1,224,224)[0], cmap='gray')
# plt.savefig("gozde_temp_images/outputt.png")

######### CNN ARCHITECTURE

import torch.nn as nn
import torch.nn.functional as F

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}],{} train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, "last_lr: {:.5f},".format(result['lrs'][-1]) if 'lrs' in result else '', 
            result['train_loss'], result['val_loss'], result['val_acc']))
            
 # --------------------------------------------------------------------------#
 
from torchvision import models

class MyModel(nn.Module):
    def __init__(self, weight=1):
        super().__init__()

        self.model = torchvision.models.resnet18(pretrained=True)
        # Change conv1 from 3 to 1 input channels
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=8)
        
    def forward(self, X):
    
        X = self.model(X)
        return X

    
model_gmd = MyModel()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_gmd.parameters(), lr=0.005)

epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

#########

class RpsClassifier(nn.Module):
    def __init__(self):
        super(RpsClassifier, self).__init__()
        self.block1 = self.conv_block(c_in=1, c_out=256, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=256, c_out=128, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=128, c_out=64, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=32, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        return x
    
    def conv_block(self, c_in, c_out, dropout, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block
    
    
model = RpsClassifier()
# model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for e in tqdm(range(1, 11)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for batch in train_loader:
        X_train_batch, y_train_batch = batch['image'], batch['labels']
        optimizer.zero_grad()
        y_train_pred = model(X_train_batch).squeeze()
        print(y_train_pred.shape)
        train_loss = criterion(y_train_pred.view(8,-1), y_train_batch)
        # train_acc = multi_acc(y_train_pred, y_train_batch)
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()
        # train_epoch_acc += train_acc.item()
        break