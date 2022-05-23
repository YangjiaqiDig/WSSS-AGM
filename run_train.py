import logging
import os
import time
from collections import Counter

from pytorch_grad_cam import GradCAM
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


DEVICE_NR = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NR
logging.basicConfig(level=logging.DEBUG)

from gan_inference import load_gan_model
from dataset import (OCTDataset, train_transform, valid_transform)
from cam import save_cam_results, refine_input_by_cam, refine_input_by_background_cam, get_pseudo_label
from models import MultiTaskModel, CAM_Net, U_Net
from oct_utils import OrgLabels, calculate_metrics
from options import Configs


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

transform_norml = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
def train_once(args, epoch, trainloader, models, optimizer, train_subsampler, gan_pretrained):
    cam_model, seg_model = models['cam'], models['seg']
    cam_optimizer, seg_optimizer = optimizer['cam'], optimizer['seg']
    loss_cam = nn.BCELoss()#nn.BCEWithLogitsLoss()#
    loss_seg = nn.CrossEntropyLoss()
    cam_model.train()
    target_layers = [cam_model.multi_task_model.base_model[-1][-1]]
    cam = GradCAM(model=cam_model, use_cuda=args.device, target_layers=target_layers)
    trainloader.dataset.set_use_train_transform(True)
    for batch, data in tqdm(enumerate(trainloader), total = int(len(train_subsampler)/trainloader.batch_size)):
        image, labels = data["image"].to(args.device), data["labels"].to(args.device)
        updated_image = image.clone()
        if (epoch + 1) > args.refine_epoch_point or args.continue_train:
            if (epoch + 1) > (args.refine_epoch_point + args.n_refine_background) or args.continue_train:
                updated_image = refine_input_by_cam(args, cam_model, updated_image, cam)
            else:
                updated_image = refine_input_by_background_cam(args, cam_model, updated_image, cam)
        if args.input_gan:
            input_for_gan = transform_norml(updated_image)
            gan_tensor = gan_pretrained.inference(input_for_gan)
            updated_image = torch.cat((image, gan_tensor), dim=1)

        cam_optimizer.zero_grad()
        outputs = cam_model(updated_image)
        loss_train = loss_cam(outputs, labels)
        loss_train.backward()
        cam_optimizer.step()
    
    cam_model.eval()
    total_acc = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
    total_loss = 0
    for batch, data in enumerate(trainloader):
        image, labels = data["image"].to(args.device), data["labels"].to(args.device)
        updated_image = image.clone()
        if (epoch + 1) > args.refine_epoch_point or args.continue_train:
            if (epoch + 1) > (args.refine_epoch_point + args.n_refine_background) or args.continue_train:
                updated_image = refine_input_by_cam(args, cam_model, updated_image, cam)
            else:
                updated_image = refine_input_by_background_cam(args, cam_model, updated_image, cam)
        if args.input_gan:
            input_for_gan = transform_norml(updated_image)
            gan_tensor = gan_pretrained.inference(input_for_gan)
            updated_image = torch.cat((image, gan_tensor), dim=1)
        with torch.no_grad():
            outputs = cam_model(updated_image)
            loss_train = loss_cam(outputs, labels)
            total_loss += loss_train.cpu().item()
            batch_accuracies_metrics = calculate_metrics(outputs, labels)
            total_acc += Counter(batch_accuracies_metrics)
    train_acc_epoch, train_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc.items()}, total_loss / (batch + 1)
    print('Epoch', str(epoch + 1), 'Train loss:', train_loss_epoch, "Train acc", train_acc_epoch)
    return train_loss_epoch, train_acc_epoch

def valid_once(args, fold, epoch, testloader, models, test_subsampler, gan_pretrained):
    cam_model, seg_model = models['cam'], models['seg']
    cam_model.eval()
    loss_func = nn.BCELoss()
    # Evaluationfor this fold
    total_acc_val = Counter({'acc': 0, 'f1m': 0, 'f1mi': 0})
    total_loss_val = 0
    target_layers = [cam_model.multi_task_model.base_model[-1][-1]]
    cam = GradCAM(model=cam_model, use_cuda=args.device, target_layers=target_layers)
    testloader.dataset.set_use_train_transform(False)
    for batch, data in tqdm(enumerate(testloader), total=int(len(test_subsampler) / testloader.batch_size)):
        image, labels = data["image"].to(args.device), data["labels"].to(args.device)        
        updated_image = image.clone()
        if (epoch + 1) > args.refine_epoch_point or args.continue_train:
            if (epoch + 1) > (args.refine_epoch_point + args.n_refine_background) or args.continue_train:
                updated_image = refine_input_by_cam(args, cam_model, updated_image, cam)
            else:
                updated_image = refine_input_by_background_cam(args, cam_model, updated_image, cam)
        if args.input_gan:
            input_for_gan = transform_norml(updated_image)
            gan_tensor = gan_pretrained.inference(input_for_gan)
            updated_image = torch.cat((image, gan_tensor), dim=1)
        outputs = cam_model(updated_image)
        # maybe only for args.n_epochs in first condition
        if (epoch + 1) % 5 == 0 or args.continue_train or (epoch + 1) > args.refine_epoch_point: 
            params = {'args': args, 'epoch': epoch, 'model': cam_model, 'fold': fold, 'inputs': data, 'batch_preds': outputs, 'refined': updated_image}
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
    num_input_channel = dataset[0]['image'].shape[0] * 2 if args.input_gan else dataset[0]['image'].shape[0]
    num_of_epochs = args.num_iteration if args.continue_train else args.n_epochs
    gan_pretrained = False
    if args.input_gan:
        with torch.no_grad():
            path = "{}/netG.pth".format(args.model_gan)
            gan_pretrained_dict = torch.load(path)['state_dict'] #, map_location='cpu'
            gan_pretrained = load_gan_model(gan_pretrained_dict, args.device)
            print(f' Loaded Pretained GAN weights from {path}.')
            
    start = time.time()
    # K-fold Cross Validation model evaluation
    total_train_loss, total_val_loss, total_train_acc_matrix, total_val_acc_matrix = {}, {}, {}, {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        logging.info(f'---------FOLD {fold}--------')
        # model = MultiTaskModel(backbone, num_class, num_input_channel)
        shared_model = MultiTaskModel(backbone, num_input_channel)
        cam_model = CAM_Net(shared_model, num_class)
        seg_model = U_Net(shared_model, num_class)

        if args.continue_train:
            checkpoint = torch.load('{0}/fold-{1}/weights/25.pwf'.format(args.check_point, fold))   
            print('Loading pretrained model from checkpoint {0}/fold-{1}/weights/25.pwf'.format(args.check_point, fold)) 
            cam_model.load_state_dict(checkpoint['state_dict'])
        cam_model = cam_model.cuda() if args.device == "cuda" else cam_model
        seg_model = seg_model.cuda() if args.device == "cuda" else seg_model
        
        cam_optimizer = optim.SGD(cam_model.parameters(), lr=args.lr, momentum=0.9)
        seg_optimizer = optim.Adam(seg_model.parameters(), lr=args.lr)
        
        models = {"cam": cam_model, "seg": seg_model}
        optimizer = {"cam": cam_optimizer, "seg": seg_optimizer}
        
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
            train_loss, train_acc_matrix = train_once(args, epoch, trainloader, models, optimizer, train_subsampler, gan_pretrained)
            mark_epoch = epoch + 25 if args.continue_train else epoch
            total_train_loss[mark_epoch] = total_train_loss[mark_epoch] + train_loss if mark_epoch in total_train_loss else 0
            total_train_acc_matrix[mark_epoch] =  {k: v + total_train_acc_matrix[mark_epoch][k] for k, v in train_acc_matrix.items()} if mark_epoch in total_train_acc_matrix else train_acc_matrix
            
            if (epoch + 1) % 5 == 0 or args.continue_train or (epoch + 1) > args.refine_epoch_point:
                valid_loss, valid_acc_matrxi = valid_once(args, fold, epoch, testloader, models, test_subsampler, gan_pretrained)
                total_val_loss[mark_epoch] = total_val_loss[mark_epoch] + valid_loss if mark_epoch in total_val_loss else 0
                total_val_acc_matrix[mark_epoch] =  {k: v + total_val_acc_matrix[mark_epoch][k] for k, v in valid_acc_matrxi.items()} if mark_epoch in total_val_acc_matrix else valid_acc_matrxi
            
        ## Save model for last epoch
        save_path = f'./{args.save_folder}/fold-{fold}/weights'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save({
            'epoch': epoch,
            'args': args,
            'state_dict': cam_model.state_dict(),
            'optimizer': cam_optimizer.state_dict(),
        }, save_path + "/{0}.pwf".format(epoch + 1)) 
        
    for epoch_nbr in total_train_loss:
        tb.add_scalar('Loss/Train', total_train_loss[epoch_nbr] / args.k_folds, epoch_nbr+1)
        for acc_type in ['acc', 'f1m']:
            tb.add_scalar("Train Accuracy/{}".format(acc_type), total_train_acc_matrix[epoch_nbr][acc_type] / args.k_folds, epoch_nbr)
        for label_type in OrgLabels:
            tb.add_scalar("Train Class Acc/{}".format(label_type), total_train_acc_matrix[epoch_nbr][label_type] / args.k_folds, epoch_nbr)
    for epoch_nbr in total_val_loss:
        tb.add_scalar('Loss/Valid', total_val_loss[epoch_nbr] / args.k_folds, epoch_nbr+1)
        for acc_type in ['acc', 'f1m']:
            tb.add_scalar("Val Accuracy/{}".format(acc_type),  total_val_acc_matrix[epoch_nbr][acc_type] / args.k_folds, epoch_nbr)
        for label_type in OrgLabels:
            tb.add_scalar("Val Class Acc/{}".format(label_type), total_val_acc_matrix[epoch_nbr][label_type] / args.k_folds, epoch_nbr)
    tb.close()
    print('final running time:', time.time() - start)


if __name__ == "__main__":
    args = Configs().parse()
    tb = SummaryWriter('runs/{}'.format(args.save_folder.split('/')[-1]))
    train(args)
