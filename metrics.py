"""Evaluation Metrics for Semantic Segmentation"""
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from utils import CAT_LIST, OrgLabels, convert_duke_labels, convert_resc_labels, get_num_classes
from PIL import Image
import os
import argparse
import glob

'''Classification: accuracy + f1'''
def calculate_classification_metrics(outputs, labels):
    predicted = np.round(outputs.cpu().detach().numpy())
    gt = labels.cpu().detach().numpy()

    acc, f1m, f1mi, class_acc = 0, 0, 0, [0]*len(OrgLabels) 
    for i, pred in enumerate(predicted):
        # fix the accuracy
        acc +=  accuracy_score(gt[i],pred)
        f1m += f1_score(gt[i],pred,average = 'macro', zero_division=1)
        f1mi += f1_score(gt[i],pred,average = 'micro', zero_division=1)
        class_acc += (gt[i]==pred) * 1
    class_acc = class_acc / len(outputs)
    res_dic = {'acc': acc / len(outputs), 'f1m': f1m / len(outputs), 'f1mi': f1mi / len(outputs)}
    for i in range(len(OrgLabels)):
        res_dic[OrgLabels[i]] = class_acc[i]
        
    return res_dic

'''Classification: input probabily'''
def calculate_roc(outputs, labels):
    # [batch, class]
    predicted = np.array(outputs)
    gt = np.array(labels)
    print(gt.sum(axis = 0), np.round(predicted).sum(axis = 0), gt.shape)
    # drop background class as they are all 1 then eroor roc
    if OrgLabels[-1] == 'BackGround':
        gt = gt[:, :-1]
        predicted = predicted[:, :-1]
    roc_class = roc_auc_score(gt, predicted, average=None)
    roc_avg = roc_auc_score(gt, predicted, average='weighted')
    if get_num_classes() == 1:
        if 'BackGround' in OrgLabels:
            res_dict = {OrgLabels[0]: roc_class, OrgLabels[1]: 1}
        else: res_dict = {OrgLabels[0]}
        return roc_avg, res_dict
    res_dict = {}
    for i in range(len(OrgLabels)):
        if i == len(OrgLabels) - 1:
            res_dict[OrgLabels[i]] = 1
        else:
            res_dict[OrgLabels[i]] = roc_class[i]
    return roc_avg, res_dict

'''sem seg Dice: input probbaly map'''


'''sem seg f1'''
def calculate_F1(pred_path, gt_path, numofclass):
    TPs = [0] * numofclass
    FPs = [0] * numofclass
    FNs = [0] * numofclass
    ims = os.listdir(pred_path)
    for im in ims:
        pred = np.asarray(Image.open(os.path.join(pred_path, im)))
        gt = np.asarray(Image.open(os.path.join(gt_path, im)))
        for k in range(numofclass):
            TPs[k] += np.sum(np.logical_and(pred == k, gt == k))
            FPs[k] += np.sum(np.logical_and(pred == k, gt != k))
            FNs[k] += np.sum(np.logical_and(pred != k, gt == k))
 
    f1_score = TPs / (TPs + (FPs + FNs)/2 + 1e-7)
    f1_score = sum(f1_score) / numofclass
    return f1_score

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))
    # import pdb; pdb.set_trace()

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }

def record_score(score, save_path):
    score_list = []
    for i in range(get_num_classes()+1):
        score_list.append(score['Class IoU'][i])
        aveJ = score['Mean IoU']
    with open('iou_collections/{}_iou.txt'.format(save_path), 'w') as f:
        for num, cls_iou in enumerate(score_list):
            print('class {:2d} {:12} IU {:.2f}'.format(num, CAT_LIST[num], round(cls_iou, 3)))
            f.write('class {:2d} {:12} IU {:.2f}'.format(num, CAT_LIST[num], round(cls_iou, 3)) + '\n')
        print('meanIOU: ' + str(aveJ) + '\n')
        f.write('meanIOU: ' + str(aveJ) + '\n')    
        f.write('pixelAcc: ' + str(score['Pixel Accuracy']) + '\n' + 'meanAcc: ' + str(score['Mean Accuracy']) + '\n') 

'''mIoU'''
categories = ['background', 'Fluid'] #['background', 'SRF', 'PED']
def do_python_eval(predict_folder, gt_folder, name_list, input_type='png', threshold=1.0):
    pred_list = []
    gt_list = []

    for name in name_list:
        if input_type == 'png':
            predict_file = os.path.join(predict_folder,'%s.png'%name)
            predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
        elif input_type == 'npy':
            predict_file = os.path.join(predict_folder,'%s.npy'%name)
            predict_dict = np.load(predict_file, allow_pickle=True).item()
            h, w = list(predict_dict.values())[0].shape
            tensor = np.zeros((len(categories),h,w),np.float32)
            for key in predict_dict.keys():
                tensor[key+1] = predict_dict[key]
            tensor[0,:,:] = threshold 
            predict = np.argmax(tensor, axis=0).astype(np.uint8)
            # predict = cv2.resize(predict, dsize=(512, 512))
            # print(predict.shape, np.unique(predict))
        pred_list.append(predict)
        
        '''Duke only'''
        duke_gt_name = name.split('+')[-1] + '.png'
        resc_gt_name = '%s.bmp'%name
        gt_file = os.path.join(gt_folder,duke_gt_name)
        gt = np.array(Image.open(gt_file))
        # gt = cv2.resize(gt, dsize=(512, 512))
        # import pdb; pdb.set_trace()
        '''for RESC'''
        # gt = convert_resc_labels(gt)
        gt = convert_duke_labels(gt)

        # print(gt.shape, np.unique(gt))
        gt_list.append(gt)
    return pred_list, gt_list
 

if __name__ == '__main__':
    # python metrics.py  --type npy --curve True
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='baseline_models/SEAM/duke/duke_cam/*', type=str)
    # parser.add_argument("--list", default='baseline_models/WSMIS/irn/result/cam/*', type=str)
    parser.add_argument("--predict_dir", default='baseline_models/SEAM/duke/duke_cam', type=str)
    # parser.add_argument("--predict_dir", default='baseline_models/WSMIS/irn/result/cam', type=str)
    parser.add_argument("--gt_dir", default='datasets/2015_BOE_Chiu/segment_annotation/labels', type=str)
    # parser.add_argument("--gt_dir", default='datasets/RESC/valid/label_images', type=str)
    
    # parser.add_argument('--type', default='npy', choices=['npy', 'png'], type=str)
    parser.add_argument('--type', default='npy', choices=['npy', 'png'], type=str)
    
    parser.add_argument('--t', default=0.6, type=float) # 0.3 is the highest mIoU by curve
    parser.add_argument('--curve', default=True, type=bool)
    parser.add_argument('--log_dir', default='seam_duke', type=str)
    args = parser.parse_args()

    if args.type == 'npy':
        assert args.t is not None or args.curve
    # df = pd.read_csv(args.list, names=['filename'])
    name_list = [pth.split('/')[-1].split('.')[0] for pth in glob.glob(args.list)] #df['filename'].values
    import pdb; pdb.set_trace()
    print('Disease size: ', len(name_list))
    if not args.curve:
        pred_list, gt_list = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.type, args.t)
        score = scores(gt_list, pred_list, n_class=3)
        print(score)
        record_score(score, args.log_dir)
    else:
        l = []
        for i in range(60, 100):
            t = i/100.0
            pred_list, gt_list = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.type, t)
            score = scores(gt_list, pred_list, n_class=len(categories))
            print(score)
            record_score(score, args.log_dir)
            l.append(score['Mean IoU'])
            print('%d/60 background score: %.3f\tmIoU: %.3f%%'%(i, t, score['Mean IoU']))

# if __name__ == "__main__":
#     x = np.array([[1., 1., 1., 0., 0., 1], [1., 1., 1., 0., 0., 1]])
#     y = np.array([[0., 1., 1., 1., 0., 1], [0., 1., 1., 1., 0., 1]])
#     print(calculate_classification_metrics(x, y))
#     x = torch.empty(0,2)
#     y = torch.tensor([[3, 4], [3, 4]])
    
#     for i in range(0,4):
#         x = torch.cat((x, y))
#     print(x)

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