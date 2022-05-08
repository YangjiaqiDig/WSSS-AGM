
from collections import Counter
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from cam import save_cam_results
from dataset import (OCTDataset, normal_transform, train_transform, valid_transform)
from model import MultiTaskModel
from oct_utils import OrgLabels, calculate_metrics
from options import Configs
from run_train import network_class

DEVICE_NR = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NR
def inference(args):
    if args.device == "cuda":
        print("Number of GPUs: ", torch.cuda.device_count(), "Device Nbr: ", DEVICE_NR)
        
    dataset = OCTDataset(args, transform_train=train_transform(args.is_size), transform_val=valid_transform(args.is_size))
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
        save_cam_results(params, is_inference=True)
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
    inference(args)