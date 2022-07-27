import os
import torch
from options import Configs

OrgLabels = Configs().get_labels()#['SRF', 'IRF', 'EZ', 'HRD',  'RPE', 'BackGround']

def save_models(args, epoch, cam_model, cam_optimizer, is_best=False):
    save_path = f'./{args.save_folder}/weights'
    save_name = 'best' if is_best else epoch + 1
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
        'epoch': epoch,
        'args': args,
        'state_dict': cam_model.state_dict(),
        'optimizer': cam_optimizer.state_dict(),
    }, save_path + "/{0}.pwf".format(save_name)) 

def save_tensorboard(tb, loss_dict, mark_epoch, include_valid):
    tb.add_scalar('Loss/Train', loss_dict['total_train_loss'], mark_epoch+1)
    tb.add_scalar("ROC/Train",  loss_dict['total_train_roc'], mark_epoch+1)
    for acc_type in ['acc', 'f1m']:
        tb.add_scalar("Train Accuracy/{}".format(acc_type), loss_dict['total_train_acc_matrix'][acc_type], mark_epoch+1)
    for label_type in OrgLabels:
        tb.add_scalar("Train Class Acc/{}".format(label_type), loss_dict['total_train_acc_matrix'][label_type], mark_epoch+1)
        tb.add_scalar("Train Class ROC/{}".format(label_type), loss_dict['total_train_roc_matrix'][label_type], mark_epoch+1)
    if include_valid:
        tb.add_scalar('Loss/Valid', loss_dict['total_val_loss'], mark_epoch+1)
        tb.add_scalar("ROC/Valid",  loss_dict['total_val_roc'], mark_epoch+1)
        for acc_type in ['acc', 'f1m']:
            tb.add_scalar("Val Accuracy/{}".format(acc_type),  loss_dict['total_val_acc_matrix'][acc_type], mark_epoch+1)
        for label_type in OrgLabels:
            tb.add_scalar("Val Class Acc/{}".format(label_type), loss_dict['total_val_acc_matrix'][label_type], mark_epoch+1)
            tb.add_scalar("Val Class ROC/{}".format(label_type), loss_dict['total_val_roc_matrix'][label_type], mark_epoch+1)

    tb.close()