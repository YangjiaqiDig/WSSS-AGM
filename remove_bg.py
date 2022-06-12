import glob, os
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from preprocess import background_mask
from torchvision import transforms
import torchvision.utils as vutils
from tqdm import tqdm
import cv2

from unet import U_Net
DEVICE_NR = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NR


class OUR_DATASET(Dataset): 
    def __init__(self, data_type):
        train_normal = glob.glob("our_dataset/original/train/NORMAL*") + glob.glob("oct_kaggle/train/0.normal/NORMAL*")
        self.file_list = {'train': train_normal, 'test': glob.glob("our_dataset/original/test/*")}
        self.data_type = data_type
        self.transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(15)),
                transforms.ToTensor(),
            ])
        self.transform_val = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
    def __getitem__(self, idx):
        data_path = sorted(self.file_list[self.data_type])[idx]
        image_arr = np.array(cv2.imread(data_path))
        origin_image = self.transform_val(image_arr)
        normalized_image = image_arr.copy()
        normalized_image[normalized_image > 250] = 0
        normalized_image = Image.fromarray(normalized_image)
        # image = Image.open(data_path)
        normalized_image = ImageEnhance.Contrast(normalized_image).enhance(2)
        normalized_image = np.array(normalized_image) 
        
        # if (normalized_image.ndim == 2):
        #     normalized_image = np.repeat(normalized_image[..., np.newaxis], 3, -1)
        if self.data_type == 'train':
            image_tensor = self.transform_train(normalized_image)
        else: image_tensor = self.transform_val(normalized_image)
        labels = background_mask(normalized_image)
        labels_tensor = self.transform_val(labels)
        return {'image': image_tensor, 'labels': labels_tensor.squeeze(0), 'path': data_path, 'orig': origin_image}
    def __len__(self):
        return len(self.file_list[self.data_type])
    
def save_models(epoch, model, optimizer):
    save_path = f'outputs_background/weights3'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, save_path + "/{0}.pwf".format(epoch + 1))     
    
dataset_train = OUR_DATASET(data_type='train')
dataset_test = OUR_DATASET(data_type='test')
trainloader = torch.utils.data.DataLoader(
                        dataset_train, 
                        num_workers=8,
                        batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(
                dataset_test,
                num_workers=8,
                batch_size=1, shuffle=False)
model = U_Net()
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-5)
num_epochs = 200
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)

for epoch in range(num_epochs):
    total_loss = []
    val_total_loss = []
    print('-------------', epoch)
    model.train()
    for batch, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        img, labels, path = data['image'], data['labels'].long(), data['path']
        output = model(img.to('cuda'))
        loss = criterion(output, labels.to('cuda'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.append(loss.data.item())
    lr_scheduler.step()
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, sum(total_loss) / len(total_loss)))
    
    if (epoch +1) % 20 == 0:
        model.eval()
        for batch, data in tqdm(enumerate(testloader), total=len(testloader)):
            img, labels, path = data['image'], data['labels'].long(), data['path'][0]
            output = model(img.to('cuda'))
            with torch.no_grad():
                loss = criterion(output, labels.to('cuda'))
                val_total_loss.append(loss.data.item())
                img_path = '{}'.format(path.split('/')[-1].split('.')[0])
                save_path = os.path.join('outputs_background/images3', img_path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_image = torch.cat((img[:,0], output[:,1].cpu(), labels), 0)
                # vutils.save_image(save_image.unsqueeze(1), save_path + '/{}.jpg'.format(epoch + 1), normalize=True, scale_each=True)
           
        # save_models(epoch, model, optimizer)   
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, sum(val_total_loss) / len(val_total_loss)))
         
       
       
