import logging
import os
import time
from collections import Counter
from random import shuffle
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

from cam import save_cam_results
from dataset import (OCTDataset, normal_transform)
from model import MultiClassModel, MultiTaskModel
from oct_utils import OrgLabels, calculate_metrics
from options import Configs
import cv2
import torchvision.utils as vutils


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

def refine_input_by_cam(model, image, cam):
    outputs = model(image)
    pred_classes = (outputs > 0.5) * 1 # [batch, cls]
    batch_cam_masks = []
    for cls in range(len(OrgLabels)):
        targets = [ClassifierOutputTarget(cls)]
        grayscale_cam = np.array([cam(input_tensor=each_image.unsqueeze(0),targets=targets,eigen_smooth=False, aug_smooth=True) for each_image in image])
        grayscale_cam = np.repeat(grayscale_cam, 3, 1)
        grayscale_tensor = torch.from_numpy(grayscale_cam).to(args.device)
        batch_cam_masks.append(grayscale_tensor) # cls, [batch, 3, w, h]
        
    for idx_input in range(len(pred_classes)):
        curr_cam_mask = [batch_cam_mask[idx_input] for batch_cam_mask in batch_cam_masks] # cls, [3, w, h]
        curr_preds = pred_classes[idx_input] # cls
        target_classes_cam = [class_cam * curr_preds[l] for l, class_cam in enumerate(curr_cam_mask[:-1])]
        sum_masks = sum(target_classes_cam)
        min, max = sum_masks.min(), sum_masks.max()
        sum_masks.add_(-min).div_(max - min + 1e-5) # does norm -> multiply order matter?
        background_mask = curr_cam_mask[-1]
        
        updated_input_tensor = image
        updated_input_tensor[idx_input, :3,] = sum_masks * background_mask * image[idx_input, :3,]
        return updated_input_tensor

def train_once(args, epoch, trainloader, model, optimizer, train_subsampler):
    loss_func = nn.BCELoss()#nn.BCEWithLogitsLoss()#
    model.train()
    target_layers = [model.base_model.layer4[-1]]
    cam = GradCAM(model=model, use_cuda=args.device, target_layers=target_layers)
    for batch, data in tqdm(enumerate(trainloader), total = int(len(train_subsampler)/trainloader.batch_size)):
        image, labels = data["image"].to(args.device), data["labels"].to(args.device)
        updated_image = image
        if (epoch + 1) > args.refine_epoch_point:
            updated_image = refine_input_by_cam(model, updated_image, cam)
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
        updated_image = image
        if (epoch + 1) > args.refine_epoch_point:
            updated_image = refine_input_by_cam(model, updated_image, cam)
        with torch.no_grad():
            outputs = model(updated_image)
            loss_train = loss_func(outputs, labels)
            total_loss += loss_train.cpu().item()
            batch_accuracies_metrics = calculate_metrics(outputs, labels)
            total_acc += Counter(batch_accuracies_metrics)
    train_acc_epoch, train_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc.items()}, total_loss / (batch + 1)
    print('Epoch', str(epoch + 1), 'Train loss:', train_loss_epoch, "Train acc", train_acc_epoch)
    return train_loss_epoch

def valid_once(args, fold, epoch, testloader, model, optimizer, test_subsampler):
    save_path = f'./{args.save_folder}/fold-{fold}'
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
    for batch, data in tqdm(enumerate(testloader), total=int(len(test_subsampler) / testloader.batch_size)):
        image, labels = data["image"].to(args.device), data["labels"].to(args.device)
        updated_image = image
        if (epoch + 1) > args.refine_epoch_point:
            updated_image = refine_input_by_cam(model, updated_image, cam)
        outputs = model(updated_image)
        # if (epoch + 1) % args.n_epochs == 0:
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
    return valid_loss_epoch

def train(args):
    if args.device == "cuda":
        print("Number of GPUs: ", torch.cuda.device_count(), "Device Nbr: ", DEVICE_NR)
    kfold = KFold(n_splits=args.k_folds, shuffle=False)
    dataset = OCTDataset(args, transform=normal_transform(args.is_size))
    start = time.time()
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        logging.info(f'---------FOLD {fold}--------')
        # # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        # train_dataset = OCTDataset(args.root_dirs, transform=train_transform())
        # test_dataset = OCTDataset(args.root_dirs, transform=valid_transform())
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        num_workers=8,
                        batch_size=args.train_batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        num_workers=8,
                        batch_size=args.valid_batch_size, sampler=test_subsampler, shuffle=False)
        backbone = network_class(args)
        num_class = len(OrgLabels)
        num_input_channel = dataset[0]['image'].shape[0]
        model = MultiTaskModel(backbone, num_class, num_input_channel)
        model = model.cuda() if args.device == "cuda" else model
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        for epoch in range(0, args.n_epochs):
            train_loss = train_once(args, epoch, trainloader, model, optimizer, train_subsampler)
            if (epoch + 1) % 5 == 0:
                valid_loss = valid_once(args, fold, epoch, testloader, model, optimizer, test_subsampler)

    print('final running time:', time.time() - start)

def train_multi_class(args):
    if args.device == "cuda":
        print("Number of GPUs: ", torch.cuda.device_count(), "Device Nbr: ", DEVICE_NR)
    kfold = KFold(n_splits=args.k_folds, shuffle=False)
    dataset = OCTDataset(args, transform=normal_transform(args.is_size))
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        logging.info(f'---------FOLD {fold}--------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        num_workers=8,
                        batch_size=args.train_batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        num_workers=8,
                        batch_size=args.valid_batch_size, sampler=test_subsampler, shuffle=False)
        backbone = network_class(args)
        num_class = len(OrgLabels)
        num_input_channel = dataset[0]['image'].shape[0]
        model_1 = MultiTaskModel(backbone, num_class, num_input_channel)
        checkpoint = torch.load('{0}/fold-{1}/25.pwf'.format(args.save_folder, fold))   
        print('Loading pretrained model from checkpoint {0}/fold-{1}/25.pwf'.format(args.save_folder, fold)) 
        model_1.load_state_dict(checkpoint['state_dict'])
        model_1 = model_1.cuda() if args.device == "cuda" else model_1
        model_1.eval()
        target_layers = [model_1.base_model.layer4[-1]]
        cam = GradCAM(model=model_1, use_cuda=args.device, target_layers=target_layers)
        for batch, data in tqdm(enumerate(trainloader), total=int(len(dataset) / trainloader.batch_size)):
            image, labels = data["image"].to(args.device), data["labels"].to(args.device)
            outputs = model_1(image)
            pred_classes = (outputs > 0.5) * 1 # [batch, cls]
            print(pred_classes, labels)
            batch_cam_masks = []
            for cls in range(len(OrgLabels)):
                targets = [ClassifierOutputTarget(cls)]
                grayscale_cam = np.array([cam(input_tensor=each_image.unsqueeze(0),targets=targets,eigen_smooth=False, aug_smooth=True) for each_image in image])
                grayscale_cam = np.repeat(grayscale_cam, 3, 1)
                grayscale_tensor = torch.from_numpy(grayscale_cam).to(args.device)
                batch_cam_masks.append(grayscale_tensor) # cls, [batch, 3, w, h]
                
                cam_soft_mask = grayscale_tensor * image[:, :3,]
                # print(cam_soft_mask.shape, grayscale_cam)
                vutils.save_image(cam_soft_mask,'soft_cam_apply/enhance_{0}.jpg'.format(cls), normalize=True)
                # vutils.save_image(cam_soft_mask,'soft_cam_apply/enhance2_{0}.jpg'.format(cls), normalize=True, scale_each=True)
                
            for idx_input in range(len(pred_classes)):
                curr_cam_mask = [batch_cam_mask[idx_input] for batch_cam_mask in batch_cam_masks] # cls, [3, w, h]
                curr_preds = pred_classes[idx_input] # cls
                target_classes_cam = [class_cam * curr_preds[l] for l, class_cam in enumerate(curr_cam_mask[:-1])]
                sum_masks = sum(target_classes_cam)
                min, max = sum_masks.min(), sum_masks.max()
                sum_masks.add_(-min).div_(max - min + 1e-5) # does norm -> multiply order matter?
                background_mask = curr_cam_mask[-1]
                
                updated_input_tensor = image
                updated_input_tensor[idx_input, :3,] = sum_masks * background_mask * image[idx_input, :3,]
                # updated_input_tensor.add_(-min).div_(max - min + 1e-5)
                print(len(target_classes_cam), updated_input_tensor.shape)
                vutils.save_image(target_classes_cam,'soft_cam_apply/filter_{0}.jpg'.format(idx_input), normalize=True)
                vutils.save_image(sum_masks,'soft_cam_apply/norm_{0}.jpg'.format(idx_input), normalize=True)
                vutils.save_image(updated_input_tensor.reshape(-1,3,256, 256),'soft_cam_apply/updated_{0}.jpg'.format(idx_input), normalize=True, scale_each=True)
                import pdb; pdb.set_trace()


def inference(args):
    if args.device == "cuda":
        print("Number of GPUs: ", torch.cuda.device_count(), "Device Nbr: ", DEVICE_NR)
        
    dataset = OCTDataset(args, transform=normal_transform(args.is_size))
    testloader = torch.utils.data.DataLoader(
                    dataset,
                    num_workers=8,
                    batch_size=args.train_batch_size, shuffle=False)
    backbone = network_class(args)
    num_class = len(OrgLabels)
    num_input_channel = dataset[0]['image'].shape[0]
    model = MultiTaskModel(backbone, num_class, num_input_channel)
    checkpoint = torch.load('{0}{1}'.format(args.save_folder, args.check_point))   
    print('Loading pretrained model from checkpoint {0}{1}'.format(args.save_folder, args.check_point)) 
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda() if args.device == "cuda" else model
    
    model.eval()
    loss_func = nn.BCELoss()
    # Evaluation for this fold
    total_acc_val = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
    total_loss_val = 0
    for batch, data in tqdm(enumerate(testloader), total=int(len(dataset) / testloader.batch_size)):
        image, labels = data["image"].to(args.device), data["labels"].to(args.device)
        outputs = model(image)
        params = {'args': args, 'model': model, 'inputs': data, 'batch_preds': outputs}
        save_cam_results(params)
        with torch.no_grad():
            loss_val = loss_func(outputs, labels)
            total_loss_val += loss_val.cpu().item()
            batch_accuracies_metrics = calculate_metrics(outputs, labels)
            total_acc_val += Counter(batch_accuracies_metrics)

    # Print accuracy
    valid_acc_epoch, valid_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc_val.items()}, total_loss_val / (batch + 1)

    print('Val loss:', valid_loss_epoch, "Val acc:", valid_acc_epoch)
    return valid_loss_epoch

if __name__ == "__main__":
    args = Configs().parse()
    # train_multi_class(args)
    if not args.inference:
        train(args)
    else:
        inference(args)
