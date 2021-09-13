import torch.nn as nn
import torch

def get_mutil_task_loss(net_output, ground_truth):
    loss_func = nn.CrossEntropyLoss()
    emeda_loss = loss_func(net_output["edema"], ground_truth["edema"])
    dril_loss = loss_func(net_output["dril"], ground_truth["dril"])
    ez_loss = loss_func(net_output["ez"], ground_truth["ez"])
    rpe_loss = loss_func(net_output["rpe"], ground_truth["rpe"])
    total_loss = emeda_loss + dril_loss + ez_loss + rpe_loss
    return total_loss, {"edema": emeda_loss, "dril": dril_loss, "ez": ez_loss, "rpe": rpe_loss}

def calculate_metrics(preds, labels):
    batch_size = preds["edema"].shape[0]
    pred_edema = torch.argmax(preds["edema"], dim=1).cpu()
    pred_dril = torch.argmax(preds["dril"], dim=1).cpu()
    pred_ez = torch.argmax(preds["ez"], dim=1).cpu()
    pred_rpe = torch.argmax(preds["rpe"], dim=1).cpu()
    return {
        'edema': (pred_edema == labels["edema"].cpu()).sum().item() / batch_size,
        'dril': (pred_dril == labels["dril"].cpu()).sum().item() / batch_size,
        'ez': (pred_ez == labels["ez"].cpu()).sum().item() / batch_size,
        'rpe': (pred_rpe == labels["rpe"].cpu()).sum().item() / batch_size
    }
