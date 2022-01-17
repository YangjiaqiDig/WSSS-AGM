import torch
import os, time
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from model import MultiTaskModel
from sklearn.model_selection import KFold
from dataset import OCTDataset, train_transform, valid_transform
import logging
from argparse import ArgumentParser
from oct_utils import *
import copy

# logger = logging.getLogger(__file__).setLevel(logging.INFO)
logging.basicConfig(level=logging.DEBUG)

Class_Names = {'srf': 0, 'irf': 0, 'dril': 0, 'ezAtt': 0, 'ezDis': 0, 'rpe': 0, 'hrd': 0, 'rt': 0, 'qDril': 0}

def configs():
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="{0}".format("train_dr"),
                        help="Hepatic, CREMI, ISBI12, ISBI13")
    parser.add_argument("--k_folds", type=int,
                        default=5, help="k folds")
    parser.add_argument("--save_folder", type=str, default="results_dr",
                        help="Path or url of the dataset")
    parser.add_argument("--train_batch_size", type=int,
                        default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int,
                        default=1, help="Batch size for validation")
    parser.add_argument("--lr", type=float,
                        default=0.001, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--check_point", type=str, default="/model_epoch_350.pwf",
                        help="Path of the pre-trained CNN")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--device_nr", type=str, default="3", help="Device Number")
    args = parser.parse_args()

    return args

def train_once(args, epoch, trainloader, model, optimizer):
    model.train()
    for batch, data in enumerate(trainloader):
        if data["image"].shape[1] == 1:
            image = data["image"].repeat(1, 3, 1, 1)
        else: 
            image = data["image"]
        labels = data["labels"]
        optimizer.zero_grad()
        outputs = model(image.to(args.device))
        loss_train, losses_group = get_mutil_task_loss(outputs, labels, args)
        loss_train.backward()
        optimizer.step()
    model.eval()
    total_acc = copy.deepcopy(Class_Names)
    total_loss = 0
    for batch, data in enumerate(trainloader):
        if data["image"].shape[1] == 1:
            image = data["image"].repeat(1, 3, 1, 1)
        else: 
            image = data["image"]        
        labels = data["labels"]
        with torch.no_grad():
            outputs = model(image.to(args.device))
            loss_train, losses_group = get_mutil_task_loss(outputs, labels, args)
            total_loss += loss_train.cpu().item()
            batch_accuracies = calculate_metrics(outputs, labels)
            # total_acc['edema'] += batch_accuracies['edema']
            total_acc['dril'] += batch_accuracies['dril']
            # total_acc['ez'] += batch_accuracies['ez']
            total_acc['rpe'] += batch_accuracies['rpe']
            total_acc['hrd'] += batch_accuracies['hrd']
            total_acc['rt'] += batch_accuracies['rt']
            total_acc['qDril'] += batch_accuracies['qDril']
            total_acc['srf'] += batch_accuracies['srf']
            total_acc['irf'] += batch_accuracies['irf']
            total_acc['ezAtt'] += batch_accuracies['ezAtt']
            total_acc['ezDis'] += batch_accuracies['ezDis']
    train_acc_epoch, train_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc.items()}, total_loss / (batch + 1)
    print('Epoch', str(epoch + 1), 'Train loss:', train_loss_epoch, "Train acc", train_acc_epoch)

def valid_once(args, fold, epoch, testloader, model, optimizer):
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
    # Evaluationfor this fold
    total_acc_val = copy.deepcopy(Class_Names)
    total_loss_val = 0
    with torch.no_grad():
        for batch, data in enumerate(testloader):
            if data["image"].shape[1] == 1:
                image = data["image"].repeat(1, 3, 1, 1)
            else: 
                image = data["image"]
            labels = data["labels"]
            outputs = model(image.to(args.device))
            loss_val, losses_group = get_mutil_task_loss(outputs, labels, args)
            total_loss_val += loss_val.cpu().item()
            batch_accuracies = calculate_metrics(outputs, labels)
            # total_acc_val['edema'] += batch_accuracies['edema']
            total_acc_val['dril'] += batch_accuracies['dril']
            # total_acc_val['ez'] += batch_accuracies['ez']
            total_acc_val['rpe'] += batch_accuracies['rpe']
            total_acc_val['hrd'] += batch_accuracies['hrd']
            total_acc_val['rt'] += batch_accuracies['rt']
            total_acc_val['qDril'] += batch_accuracies['qDril']
            total_acc_val['srf'] += batch_accuracies['srf']
            total_acc_val['irf'] += batch_accuracies['irf']
            total_acc_val['ezAtt'] += batch_accuracies['ezAtt']
            total_acc_val['ezDis'] += batch_accuracies['ezDis']

        # Print accuracy
        valid_acc_epoch, valid_loss_epoch = {k: v  / (batch + 1) for k, v in total_acc_val.items()}, total_loss_val / (batch + 1)
        # print(f'K-FOLD CROSS VALIDATION RESULTS FOR {fold} FOLDS')
        print('Val loss:', valid_loss_epoch, "Val acc:", valid_acc_epoch)

def train(args):
    if args.device == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_nr
    kfold = KFold(n_splits=args.k_folds, shuffle=True)
    dataset = OCTDataset(args.root_dir)
    start = time.time()
    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        logging.info(f'---------FOLD {fold}--------')
        # # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        train_dataset = OCTDataset(args.root_dir, transform=train_transform())
        test_dataset = OCTDataset(args.root_dir, transform=valid_transform())
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                        train_dataset, 
                        batch_size=args.train_batch_size, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=args.valid_batch_size, sampler=test_subsampler)
        model = MultiTaskModel()
        model = model.cuda() if args.device == "cuda" else model
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        for epoch in range(0, args.n_epochs):
            train_once(args, epoch, trainloader, model, optimizer)
            if (epoch + 1) % 5 == 0:
                valid_once(args, fold, epoch, testloader, model, optimizer)
    print('final running time:', time.time() - start)

if __name__ == "__main__":
    args = configs()
    train(args)
