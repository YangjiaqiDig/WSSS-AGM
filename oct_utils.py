from turtle import pd
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

LABELS = ['srf', 'irf', 'ezAtt', 'ezDis', 'hrd', 'rpe', 'rt', 'dril']
type_color = {
    0: [0, 0, 0], # black
    1: [255, 0, 0], # srf red
    2: [0, 255, 0], # irf green
    3: [0, 0, 255], # ez blue
    4: [255, 255, 0], # hrd yellow
    5: [255, 165, 0], # rpe orange
    6: [255, 0, 255] # back ground pink
}
OrgLabels = ['SRF', 'IRF', 'EZ', 'HRD',  'RPE', 'BackGround'] # 'EZ attenuated', 'EZ disrupted', 'Retinal Traction',  'Definite DRIL']
# OrgLabels = ['SRF', 'IRF', 'HRD', 'BackGround']
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
    return res_dic

if __name__ == "__main__":
    x = np.array([[1., 1., 1., 0., 0., 1]])
    y = np.array([[0., 1., 1., 1., 0., 1]])
    print(calculate_metrics(x, y))
    # print(classification_report(y, x))
