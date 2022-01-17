import torch.nn as nn
import torch

def get_mutil_task_loss(net_output, ground_truth, args):
    loss_func = nn.CrossEntropyLoss()
    # emeda_loss = loss_func(net_output["edema"], ground_truth["edema"].to(args.device))
    dril_loss = loss_func(net_output["dril"], ground_truth["dril"].to(args.device))
    # ez_loss = loss_func(net_output["ez"], ground_truth["ez"].to(args.device))
    rpe_loss = loss_func(net_output["rpe"], ground_truth["rpe"].to(args.device))
    hrd_loss = loss_func(net_output["hrd"], ground_truth["hrd"].to(args.device))
    rt_loss = loss_func(net_output["rt"], ground_truth["rt"].to(args.device))
    qDril_loss = loss_func(net_output["qDril"], ground_truth["qDril"].to(args.device))
    srf_loss = loss_func(net_output["srf"], ground_truth["srf"].to(args.device))
    irf_loss = loss_func(net_output["irf"], ground_truth["irf"].to(args.device))
    ezAtt_loss = loss_func(net_output["ezAtt"], ground_truth["ezAtt"].to(args.device))
    ezDis_loss = loss_func(net_output["ezDis"], ground_truth["ezDis"].to(args.device))
    total_loss = srf_loss + irf_loss + dril_loss + ezAtt_loss + ezDis_loss + rpe_loss + qDril_loss + rt_loss + hrd_loss
    return total_loss, {
            # 'edema': emeda_loss,
            'dril': dril_loss,
            # 'ez': self.ez(x),
            'rpe': rpe_loss,
            'hrd': hrd_loss, 
            'rt': rt_loss, 
            'qDril': qDril_loss, 
            'srf': srf_loss, 
            'irf': irf_loss, 
            'ezAtt': ezAtt_loss, 
            'ezDis': ezDis_loss
        }

def calculate_metrics(preds, labels):
    batch_size = preds["dril"].shape[0]
    # pred_edema = torch.argmax(preds["edema"], dim=1).cpu()
    pred_dril = torch.argmax(preds["dril"], dim=1).cpu()
    # pred_ez = torch.argmax(preds["ez"], dim=1).cpu()
    pred_rpe = torch.argmax(preds["rpe"], dim=1).cpu()
    pred_hrd = torch.argmax(preds["hrd"], dim=1).cpu()
    pred_rt = torch.argmax(preds["rt"], dim=1).cpu()
    pred_qDril = torch.argmax(preds["qDril"], dim=1).cpu()
    pred_srf = torch.argmax(preds["srf"], dim=1).cpu()
    pred_irf = torch.argmax(preds["irf"], dim=1).cpu()
    pred_ezAtt = torch.argmax(preds["ezAtt"], dim=1).cpu()
    pred_ezDis = torch.argmax(preds["ezDis"], dim=1).cpu()
    
    return {
        # 'edema': (pred_edema == labels["edema"].cpu()).sum().item() / batch_size,
        'dril': (pred_dril == labels["dril"].cpu()).sum().item() / batch_size,
        # 'ez': (pred_ez == labels["ez"].cpu()).sum().item() / batch_size,
        'rpe': (pred_rpe == labels["rpe"].cpu()).sum().item() / batch_size,
        'hrd': (pred_hrd == labels["hrd"].cpu()).sum().item() / batch_size, 
        'rt': (pred_rt == labels["rt"].cpu()).sum().item() / batch_size, 
        'qDril': (pred_qDril == labels["qDril"].cpu()).sum().item() / batch_size, 
        'srf': (pred_srf == labels["srf"].cpu()).sum().item() / batch_size, 
        'irf': (pred_irf == labels["irf"].cpu()).sum().item() / batch_size, 
        'ezAtt': (pred_ezAtt == labels["ezAtt"].cpu()).sum().item() / batch_size, 
        'ezDis': (pred_ezDis == labels["ezDis"].cpu()).sum().item() / batch_size
    }
