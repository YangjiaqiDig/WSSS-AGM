import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from options import Configs
import torch

type_color = {
    0: [0, 0, 0], # black
    1: [255, 0, 0], # srf red
    2: [0, 255, 0], # irf green
    3: [0, 0, 255], # ez blue
    4: [255, 255, 0], # hrd yellow
    5: [255, 165, 0], # rpe orange
    6: [255, 0, 255] # back ground pink
}
OrgLabels = Configs().get_labels()#['SRF', 'IRF', 'EZ', 'HRD',  'RPE', 'BackGround']
# 0.8247863247863247 0.7519246396739377 0.8247863247863247
# 0.8301282051282052 0.7603308530659053 0.8301282051282052
def calculate_metrics(outputs, labels):
    predicted = np.round(outputs.cpu().detach().numpy())
    gt = labels.cpu().detach().numpy()

    acc, f1m, f1mi, class_acc = 0, 0, 0, [0]*len(OrgLabels) 
    for i, pred in enumerate(predicted):
        acc +=  accuracy_score(gt[i],pred)
        f1m += f1_score(gt[i],pred,average = 'macro', zero_division=1)
        f1mi += f1_score(gt[i],pred,average = 'micro', zero_division=1)
        class_acc += (gt[i]==pred) * 1
    class_acc = class_acc / len(outputs)
    res_dic = {'acc': acc / len(outputs), 'f1m': f1m / len(outputs), 'f1mi': f1mi / len(outputs)}
    for i in range(len(OrgLabels)):
        res_dic[OrgLabels[i]] = class_acc[i]
        
    # TODO: roc
    return res_dic

def calculate_roc(outputs, labels):
    # [batch, class]
    predicted = outputs.cpu().detach().numpy()
    gt = labels.cpu().detach().numpy()
    print(gt.sum(axis = 0), np.round(predicted).sum(axis = 0), gt.shape)
    # drop background class as they are all 1 then eroor roc
    if OrgLabels[-1] == 'BackGround':
        gt = gt[:, :-1]
        predicted = predicted[:, :-1]
    roc_class = roc_auc_score(gt, predicted, average=None)
    roc_avg = roc_auc_score(gt, predicted, average='weighted')
    res_dict = {}
    for i in range(len(OrgLabels)):
        if i == len(OrgLabels) - 1:
            res_dict[OrgLabels[i]] = 1
        else:
            res_dict[OrgLabels[i]] = roc_class[i]
    return roc_avg, res_dict

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
    
if __name__ == "__main__":
    x = np.array([[1., 1., 1., 0., 0., 1], [1., 1., 1., 0., 0., 1]])
    y = np.array([[0., 1., 1., 1., 0., 1], [0., 1., 1., 1., 0., 1]])
    print(calculate_metrics(x, y))
    import torch
    x = torch.empty(0,2)
    y = torch.tensor([[3, 4], [3, 4]])
    
    for i in range(0,4):
        x = torch.cat((x, y))
    print(x)

    # print(classification_report(y, x))
    # import numpy as np
    # from sklearn.datasets import make_multilabel_classification
    # from sklearn.multioutput import MultiOutputClassifier
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.datasets import load_iris

    # X, y = load_iris(return_X_y=True)
    # clf = LogisticRegression(solver="liblinear").fit(X, y)
    # X, y = make_multilabel_classification(random_state=0)
    # clf = MultiOutputClassifier(clf).fit(X, y)
    # # get a list of n_output containing probability arrays of shape
    # # (n_samples, n_classes)
    # y_pred = clf.predict_proba(X)
    # print(y.shape, np.array(y_pred).shape)
    # # extract the positive columns for each output
    # y_pred = np.transpose([pred[:, 1] for pred in y_pred])
    # print(roc_auc_score(y, y_pred, average='weighted'))
    # from sklearn.linear_model import RidgeClassifierCV
    # clf = RidgeClassifierCV().fit(X, y)
    # roc_auc_score(y, clf.decision_function(X), average=None)