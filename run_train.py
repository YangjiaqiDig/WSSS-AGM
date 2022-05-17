import logging
import os
import time
from collections import Counter
from turtle import pd
from pytorch_grad_cam import GradCAM
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.model_selection import KFold
from tqdm import tqdm
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from gan_inference import load_gan_model
from cam import save_cam_results
from dataset import (OCTDataset, train_transform, valid_transform)
from model import MultiTaskModel
from oct_utils import OrgLabels, calculate_metrics
from options import Configs
from torchvision import transforms

# logger = logging.getLogger(__file__).setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)
DEVICE_NR = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NR

def network_class(args):
    if args.backbone == "resnet18":
        print("Backbone: ResNet18")
        backbone = models.resnet18(pretrained=True)
    elif args.backbone == "vgg16":
        print("Backbone: VGG16")
        backbone = models.vgg16(pretrained=True)
    elif args.backbone == "resnet50":
        print("Backbone: ResNet50")
        backbone = models.resnet50(pretrained=True)
    elif args.backbone == "resnet101":
        print("Backbone: ResNet101")
        backbone = models.resnet101(pretrained=True)
    else:
        raise NotImplementedError("No backbone found for '{}'".format(args.backbone))   
    return backbone

'''
5 labels: image -> 2 / 5 (irf, ez)  -> class_prob > 0.5
[0.1, 0.8, 0.1, 0.8, 1]
cam * 5 -> prob (0-1)
Muli(cam) -> 0
Sum(cam5) -> 1
filter cam5 -> cam2 -> Sum() -> normalize
'''
def refine_input_by_cam(model, image, cam, aug_smooth=True):
    outputs = model(image)
    bacth_preds = (outputs > 0.5) * 1 # [batch, cls] -> (0,1)
    batch_cam_masks = []
    for cls in range(len(OrgLabels)):
        targets = [ClassifierOutputTarget(cls)] * len(image) # for all in batch return the current class cam
        batch_grayscale_cam = cam(input_tensor=image,targets=targets,eigen_smooth=False, aug_smooth=aug_smooth)
        grayscale_tensor = torch.from_numpy(batch_grayscale_cam).to(args.device)
        grayscale_tensor = grayscale_tensor.unsqueeze(1).repeat(1, 3, 1, 1) # extend gray to 3 channels
        batch_cam_masks.append(grayscale_tensor) # cls, [batch, 3, w, h]
    updated_input_tensor = image.clone()
    for batch_idx in range(len(bacth_preds)):
        singel_cam_masks = [batch_cam_mask[batch_idx] for batch_cam_mask in batch_cam_masks] # cls, [3, w, h]
        curr_preds = bacth_preds[batch_idx] # (cls)
        '''if predict 1, keep the class cam, else 0, turn cam to 0 black image. Except last class BackGround'''
        target_classes_cam = [class_cam * curr_preds[l] for l, class_cam in enumerate(singel_cam_masks[:-1])]
        # sum the cams for predicted classes
        sum_masks = sum(target_classes_cam)
        # normalize the above 'attention map' to 0-1
        min, max = sum_masks.min(), sum_masks.max()
        sum_masks.add_(-min).div_(max - min + 1e-5) 
        # BackGround CAM * normalized CAM * Original Image. does norm -> multiply order matter?
        background_mask = singel_cam_masks[-1] # Background CAM
        
        soft_apply = sum_masks * background_mask * image[batch_idx, :3,] # [3, w, h]
        soft_min, soft_max = soft_apply.min(), soft_apply.max()
        soft_apply.add_(-soft_min).div_(soft_max - soft_min + 1e-5)
        
        updated_input_tensor[batch_idx, :3,] = soft_apply
    return updated_input_tensor

def refine_input_by_background_cam(model, image, cam, aug_smooth=True):
    outputs = model(image)
    bacth_bg_preds = (outputs[:, -1] > 0.5) * 1 # [batch] -> (0,1)
    bg_cls = len(OrgLabels) - 1
    targets = [ClassifierOutputTarget(bg_cls)] * len(image) # for all in batch return the background cam
    batch_grayscale_cam = cam(input_tensor=image,targets=targets,eigen_smooth=False, aug_smooth=aug_smooth)
    grayscale_tensor = torch.from_numpy(batch_grayscale_cam).to(args.device)
    grayscale_tensor = grayscale_tensor.unsqueeze(1).repeat(1, 3, 1, 1) # extend gray to 3 channels [batch, 3, w, h]
    
    updated_input_tensor = image.clone() # [batch, c, w, h]
    '''Apply background cam on original image'''
    # BackGround CAM * Original Image for predicted 1 background image only.
    for batch_idx, bg_pred in enumerate(bacth_bg_preds):
        soft_apply = grayscale_tensor[batch_idx] * image[batch_idx, :3,] if bg_pred > 0 else image[batch_idx, :3] # [3, w, h]
        soft_min, soft_max = soft_apply.min(), soft_apply.max()
        soft_apply.add_(-soft_min).div_(soft_max - soft_min + 1e-5)
        updated_input_tensor[batch_idx, :3,] = soft_apply
    return updated_input_tensor

transform_norml = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
def train_once(args, epoch, trainloader, model, optimizer, train_subsampler, gan_pretrained):
    loss_func = nn.BCELoss()#nn.BCEWithLogitsLoss()#
    model.train()
    target_layers = [model.base_model.layer4[-1]]
    cam = GradCAM(model=model, use_cuda=args.device, target_layers=target_layers)
    trainloader.dataset.set_use_train_transform(True)
    for batch, data in tqdm(enumerate(trainloader), total = int(len(train_subsampler)/trainloader.batch_size)):
        image, labels = data["image"].to(args.device), data["labels"].to(args.device)
        updated_image = image.clone()
        if (epoch + 1) > args.refine_epoch_point or args.continue_train:
            if (epoch + 1) > (args.refine_epoch_point + args.n_refine_background) or args.continue_train:
                updated_image = refine_input_by_cam(model, updated_image, cam)
            else:
                updated_image = refine_input_by_background_cam(model, updated_image, cam)
        if args.input_gan:
            input_for_gan = transform_norml(updated_image)
            gan_tensor = gan_pretrained.inference(input_for_gan)
            updated_image = torch.cat((image, gan_tensor), dim=1)

        optimizer.zero_grad()
        outputs = model(updated_image)
        loss_train = loss_func(outputs, labels)
        loss_train.backward()
        optimizer.step()
    
    model.eval()
    total_acc = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
    total_loss = 0
    for batch, data in enumerate(trainloader):
        image, labels = data["image"].to(args.device), data["labels"].to(args.device)
        updated_image = image.clone()
        if (epoch + 1) > args.refine_epoch_point or args.continue_train:
            if (epoch + 1) > (args.refine_epoch_point + args.n_refine_background) or args.continue_train:
                updated_image = refine_input_by_cam(model, updated_image, cam)
            else:
                updated_image = refine_input_by_background_cam(model, updated_image, cam)
        if args.input_gan:
            input_for_gan = transform_norml(updated_image)
            gan_tensor = gan_pretrained.inference(input_for_gan)
            updated_image = torch.cat((image, gan_tensor), dim=1)
        with torch.no_grad():
            outputs = model(updated_image)
            loss_train = loss_func(outputs, labels)
            total_loss += loss_train.cpu().item()
            batch_accuracies_metrics = calculate_metrics(outputs, labels)
            total_acc += Counter(batch_accuracies_metrics)
    train_acc_epoch, train_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc.items()}, total_loss / (batch + 1)
    print('Epoch', str(epoch + 1), 'Train loss:', train_loss_epoch, "Train acc", train_acc_epoch)
    return train_loss_epoch, train_acc_epoch

def valid_once(args, fold, epoch, testloader, model, optimizer, test_subsampler, gan_pretrained):
    save_path = f'./{args.save_folder}/fold-{fold}/weights'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
        'epoch': epoch,
        'args': args,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, save_path + "/{0}.pwf".format(epoch + 1))
    model.eval()
    loss_func = nn.BCELoss()
    # Evaluationfor this fold
    total_acc_val = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
    total_loss_val = 0
    # 
    target_layers = [model.base_model.layer4[-1]]
    cam = GradCAM(model=model, use_cuda=args.device, target_layers=target_layers)
    testloader.dataset.set_use_train_transform(False)
    for batch, data in tqdm(enumerate(testloader), total=int(len(test_subsampler) / testloader.batch_size)):
        image, labels = data["image"].to(args.device), data["labels"].to(args.device)
        updated_image = image.clone
        if (epoch + 1) > args.refine_epoch_point or args.continue_train:
            if (epoch + 1) > (args.refine_epoch_point + args.n_refine_background) or args.continue_train:
                updated_image = refine_input_by_cam(model, updated_image, cam)
            else:
                updated_image = refine_input_by_background_cam(model, updated_image, cam)
        if args.input_gan:
            input_for_gan = transform_norml(updated_image)
            gan_tensor = gan_pretrained.inference(input_for_gan)
            updated_image = torch.cat((image, gan_tensor), dim=1)
        outputs = model(updated_image)
        # maybe only for args.n_epochs in first condition
        if (epoch + 1) % 5 == 0 or args.continue_train or (epoch + 1) > args.refine_epoch_point: 
            params = {'args': args, 'epoch': epoch, 'model': model, 'fold': fold, 'inputs': data, 'batch_preds': outputs, 'refined': updated_image}
            save_cam_results(params)
        with torch.no_grad():
            loss_val = loss_func(outputs, labels)
            total_loss_val += loss_val.cpu().item()
            batch_accuracies_metrics = calculate_metrics(outputs, labels)
            total_acc_val += Counter(batch_accuracies_metrics)

    # Print accuracy
    valid_acc_epoch, valid_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc_val.items()}, total_loss_val / (batch + 1)
    # print(f'K-FOLD CROSS VALIDATION RESULTS FOR {fold} FOLDS')
    print('Val loss:', valid_loss_epoch, "Val acc:", valid_acc_epoch)
    return valid_loss_epoch, valid_acc_epoch

def train(args):
    if args.device == "cuda":
        print("Number of GPUs: ", torch.cuda.device_count(), "Device Nbr: ", DEVICE_NR)
    torch.manual_seed(42)
    kfold = KFold(n_splits=args.k_folds, shuffle=False)
    dataset = OCTDataset(args, transform_train=train_transform(args.is_size), transform_val=valid_transform(args.is_size))
    backbone = network_class(args)
    num_class = len(OrgLabels)
    num_input_channel = 6#dataset[0]['image'].shape[0]
    model = MultiTaskModel(backbone, num_class, num_input_channel)
    if args.continue_train:
        checkpoint = torch.load('{0}/fold-{1}/weights/25.pwf'.format(args.save_folder, fold))   
        print('Loading pretrained model from checkpoint {0}/fold-{1}/weights/25.pwf'.format(args.save_folder, fold)) 
        model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda() if args.device == "cuda" else model
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    num_of_epochs = args.num_iteration if args.continue_train else args.n_epochs
    if args.input_gan:
        with torch.no_grad():
            path = "{}/netG.pth".format(args.model_gan)
            gan_pretrained_dict = torch.load(path)['state_dict'] #, map_location='cpu'
            gan_pretrained = load_gan_model(gan_pretrained_dict, args.device)
            print(f' Loaded Pretained GAN weights from {path}.')
            
    start = time.time()
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        logging.info(f'---------FOLD {fold}--------')
        # # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        num_workers=8,
                        batch_size=args.train_batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        num_workers=8,
                        batch_size=args.valid_batch_size, sampler=test_subsampler, shuffle=False)
        for epoch in range(0, num_of_epochs):
            train_loss, train_acc_matrix = train_once(args, epoch, trainloader, model, optimizer, train_subsampler, gan_pretrained)
            if (epoch + 1) % 5 == 0 or args.continue_train or (epoch + 1) > args.refine_epoch_point:
                valid_loss, valid_acc_matrxi = valid_once(args, fold, epoch, testloader, model, optimizer, test_subsampler, gan_pretrained)
                tb.add_scalars('Loss/Train', {'fold{}'.format(fold): train_loss}, epoch+1)
                tb.add_scalars('Loss/Valid', {'fold{}'.format(fold): valid_loss}, epoch+1)
                for acc_type in ['acc', 'f1m']:
                    tb.add_scalars("Train Accuracy/{}".format(acc_type), {'fold{}'.format(fold): train_acc_matrix[acc_type]}, epoch)
                    tb.add_scalars("Val Accuracy/{}".format(acc_type),  {'fold{}'.format(fold): valid_acc_matrxi[acc_type]}, epoch)
                for label_type in OrgLabels:
                    tb.add_scalars("Train Class Acc/{}".format(label_type), {'fold{}'.format(fold): train_acc_matrix[label_type]}, epoch)
                    tb.add_scalars("Val Class Acc/{}".format(label_type), {'fold{}'.format(fold): valid_acc_matrxi[label_type]}, epoch)
    tb.close()
    print('final running time:', time.time() - start)


if __name__ == "__main__":
    args = Configs().parse()
    tb = SummaryWriter()
    train(args)
