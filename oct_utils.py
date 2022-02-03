import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

LABELS = ['srf', 'irf', 'ezAtt', 'ezDis', 'hrd', 'rpe', 'rt', 'dril']
OrgLabels = ['SRF', 'IRF', 'EZ attenuated', 'EZ disrupted',  'HRD',  'RPE',  'Retinal Traction',  'Definite DRIL']
# 0.8247863247863247 0.7519246396739377 0.8247863247863247
# 0.8301282051282052 0.7603308530659053 0.8301282051282052
def calculate_metrics(outputs, labels):
    predicted = np.round(outputs.cpu().detach().numpy())
    gt = labels.cpu().detach().numpy()
    acc, f1m, f1mi = 0, 0, 0
    for i, pred in enumerate(predicted):
        acc +=  accuracy_score(gt[i],pred)
        f1m += f1_score(gt[i],pred,average = 'macro', zero_division=1)
        f1mi += f1_score(gt[i],pred,average = 'micro', zero_division=1)
    
    # print('f1_Macro_Score: {}'.format(f1m / len(outputs)))
    # print('f1_Micro_Score: {}'.format(f1mi / len(outputs)))
    # print('Accuracy: {}'.format(acc / len(outputs)))
    return {'acc': acc / len(outputs), 'f1m': f1m / len(outputs), 'f1mi': f1mi / len(outputs)}

if __name__ == "__main__":
    x = [1., 1., 1., 0., 1., 0., 0., 0.]
    y = [0., 1., 1., 1., 0., 0., 1., 1.]
    print(accuracy_score(x, y))