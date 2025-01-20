"""Evaluation Metrics for Semantic Segmentation"""

import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
)
from utils.utils import (
    CAT_LIST,
    OrgLabels,
    convert_duke_labels,
    convert_our_dataset_labels,
    convert_resc_labels,
    get_num_classes,
)
from PIL import Image
import argparse
import glob

"""Classification: accuracy + f1"""


def calculate_classification_metrics(outputs, labels):
    predicted = np.round(outputs.cpu().detach().numpy())
    gt = labels.cpu().detach().numpy()

    acc, f1m, f1mi, class_acc = 0, 0, 0, [0] * len(OrgLabels)
    for i, pred in enumerate(predicted):
        # fix the accuracy
        acc += accuracy_score(gt[i], pred)
        f1m += f1_score(gt[i], pred, average="macro", zero_division=1)
        f1mi += f1_score(gt[i], pred, average="micro", zero_division=1)
        class_acc += (gt[i] == pred) * 1
    class_acc = class_acc / len(outputs)
    res_dic = {
        "acc": acc / len(outputs),
        "f1m": f1m / len(outputs),
        "f1mi": f1mi / len(outputs),
    }
    for i in range(len(OrgLabels)):
        res_dic[OrgLabels[i]] = class_acc[i]

    return res_dic


"""Classification: input probabily"""


def calculate_roc(outputs, labels):
    # [batch, class]
    predicted = np.array(outputs)
    gt = np.array(labels)
    print(gt.sum(axis=0), np.round(predicted).sum(axis=0), gt.shape)
    # drop background class as they are all 1 then eroor roc
    if OrgLabels[-1] == "BackGround":
        gt = gt[:, :-1]
        predicted = predicted[:, :-1]
    roc_class = roc_auc_score(gt, predicted, average=None)
    roc_avg = roc_auc_score(gt, predicted, average="weighted")
    if get_num_classes() == 1:
        if "BackGround" in OrgLabels:
            res_dict = {OrgLabels[0]: roc_class, OrgLabels[1]: 1}
        else:
            res_dict = {OrgLabels[0]}
        return roc_avg, res_dict
    res_dict = {}
    for i in range(len(OrgLabels)):
        if i == len(OrgLabels) - 1:
            res_dict[OrgLabels[i]] = 1
        else:
            res_dict[OrgLabels[i]] = roc_class[i]
    return roc_avg, res_dict


# we use f1 macro to report the result
def calculate_classification_infer(preds_list, gt_list):
    prob_predicted = np.array(preds_list)
    sparse_predicted = np.round(prob_predicted)
    gt = np.array(gt_list)

    f1m = f1_score(gt, sparse_predicted, average="macro", zero_division=1)
    f1mi = f1_score(gt, sparse_predicted, average="micro", zero_division=1)
    acc, class_acc = 0, [0] * get_num_classes()
    for i, pred in enumerate(sparse_predicted):
        acc += accuracy_score(gt[i], pred)
        class_acc += (gt[i] == pred) * 1
    class_acc = class_acc / len(sparse_predicted)
    res_dic = {"acc": acc / len(sparse_predicted), "f1m": f1m, "f1mi": f1mi}
    for i in range(get_num_classes()):
        res_dic[OrgLabels[i]] = class_acc[i]
    # import pdb; pdb.set_trace()
    print(gt.sum(axis=0), sparse_predicted.sum(axis=0), gt.shape)
    # drop background class as they are all 1 then error roc
    roc_class = roc_auc_score(gt, prob_predicted, average=None)
    roc_avg = roc_auc_score(gt, prob_predicted, average="weighted")
    auc_dic = {"mAUC:": roc_avg}
    # import pdb; pdb.set_trace()
    for i in range(get_num_classes()):
        auc_dic[OrgLabels[i]] = roc_class[i]
    return res_dic, auc_dic


"""sem seg Dice: input probbaly map"""


"""sem seg f1"""


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

    f1_score = TPs / (TPs + (FPs + FNs) / 2 + 1e-7)
    f1_score = sum(f1_score) / numofclass
    return f1_score


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class**2,
    ).reshape(n_class, n_class)
    return hist


# micro or macro (use micro)
def Dice(y_target, y_pred, n_class):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    ru_collect = []
    ro_collect = []
    dice_score_collect = []
    macro_hist = np.zeros((n_class, n_class))
    for lt, lp in zip(y_target, y_pred):
        single_hist = _fast_hist(lt.flatten(), lp.flatten(), n_class)
        intersection = np.diag(single_hist)
        union = single_hist.sum(axis=1) + single_hist.sum(axis=0) - intersection
        dice_score = (2.0 * intersection) / (union + intersection)
        dice_score_collect.append(dice_score)
        macro_hist += single_hist

        # false negative / true positive
        fn = single_hist.sum(axis=1) - intersection
        fp = single_hist.sum(axis=0) - intersection
        if intersection[1] == 0 and 1 in lt:
            continue
        if intersection[2] == 0 and 2 in lt:
            continue
        ru = (fn + 1e-12) / (intersection + 1e-12)
        ro = (fp + 1e-12) / (intersection + 1e-12)
        ru_collect.append(ru)
        ro_collect.append(ro)

    avg_ru = np.nanmean(ru_collect, axis=0)
    avg_ro = np.nanmean(ro_collect, axis=0)
    print("ru: ", avg_ru, "ro: ", avg_ro)
    # avg_dice_per_class = sum(dice) / len(dice)
    avg_dice_per_class = np.nanmean(dice_score_collect, axis=0)
    cls_dice = dict(zip(range(n_class), avg_dice_per_class))

    macro_intersection = np.diag(macro_hist)
    macro_union = macro_hist.sum(axis=1) + macro_hist.sum(axis=0) - macro_intersection
    macro_dice_score = (2.0 * macro_intersection + 1e-6) / (
        macro_union + macro_intersection + 1e-6
    )
    # import pdb; pdb.set_trace()
    return {
        "Micro Class Dice": cls_dice,
        # 'Micro Mean Dice': np.mean(avg_dice_per_class),
        "Micro Mean Dice": np.nanmean(avg_dice_per_class),
        "Macro Class Dice": dict(zip(range(n_class), macro_dice_score)),
        "Macro Mean Dice": np.mean(macro_dice_score),
    }


def assd_score(y_target, y_pred):
    from medpy.metric.binary import assd

    score_sd = 0
    ccc = 0
    for g, p in zip(y_target, y_pred):
        # print(ccc)
        # import pdb; pdb.set_trace()
        try:
            score_sd += assd(g, p)
            ccc += 1
        except:
            # import pdb; pdb.set_trace()
            continue
    print(score_sd / ccc, ccc)
    return score_sd / ccc


def scores(label_trues, label_preds, n_class):
    macro_hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        macro_hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    # acc
    acc = np.diag(macro_hist).sum() / macro_hist.sum()
    acc_cls = np.nanmean(np.diag(macro_hist) / macro_hist.sum(axis=1))
    # iou
    macro_intersection = np.diag(macro_hist)
    macro_union = macro_hist.sum(axis=1) + macro_hist.sum(axis=0) - macro_intersection
    macro_iu = macro_intersection / macro_union
    valid = macro_hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(macro_iu[valid])
    freq = macro_hist.sum(axis=1) / macro_hist.sum()
    fwavacc = (freq[freq > 0] * macro_iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), macro_iu))
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
    for i in range(get_num_classes() + 1):
        score_list.append(score["Class IoU"][i])
        aveJ = score["Mean IoU"]
    with open("iou_collections/{}_iou.txt".format(save_path), "w") as f:
        for num, cls_iou in enumerate(score_list):
            print(
                "class {:2d} {:12} IU {:.2f}".format(
                    num, CAT_LIST[num], round(cls_iou, 3)
                )
            )
            f.write(
                "class {:2d} {:12} IU {:.2f}".format(
                    num, CAT_LIST[num], round(cls_iou, 3)
                )
                + "\n"
            )
        print("meanIOU: " + str(aveJ) + "\n")
        f.write("meanIOU: " + str(aveJ) + "\n")
        f.write(
            "pixelAcc: "
            + str(score["Pixel Accuracy"])
            + "\n"
            + "meanAcc: "
            + str(score["Mean Accuracy"])
            + "\n"
        )


"""mIoU"""


def do_python_eval(
    predict_folder, gt_folder, name_list, input_type="png", threshold=1.0
):
    pred_list = []
    gt_list = []
    for name in name_list:
        if input_type == "png":
            predict_file = os.path.join(predict_folder, "%s.png" % name)
            predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
        elif input_type == "npy":
            predict_file = os.path.join(predict_folder, "%s.npy" % name)
            predict_dict = np.load(predict_file, allow_pickle=True).item()
            # print(predict_dict.keys())
            # predict_dict_map = predict_dict['map'] if len(predict_dict['map'].shape)==3 else predict_dict['map'][:,0]
            predict_dict_map = predict_dict
            # h, w = predict_dict_map[0].shape
            h, w = list(predict_dict.values())[0].shape

            holder = np.zeros((len(categories), h, w), np.float32)
            # for wsmis only
            # for idx, key in enumerate(predict_dict['keys']): # keys
            #     if idx == 0:
            #         continue
            #     holder[key] = predict_dict_map[idx-1] #/ np.max(predict_dict_map[idx-1])
            # holder[key+1] = predict_dict_map[idx] # / np.max(predict_dict_map[idx])
            for key in predict_dict:
                holder[key + 1] = predict_dict[key]
                # holder[key] = predict_dict[key]
            holder[0, :, :] = threshold
            predict = np.argmax(holder, axis=0).astype(np.uint8)

            # predict = cv2.resize(predict, dsize=(512, 512))
            # print(predict.shape, np.unique(predict))
        pred_list.append(predict)
        if args.dataset_type == "duke":
            gt_name = re.split(r"\+|-", name)[-1] + ".png"
        elif args.dataset_type == "resc":
            gt_name = name + ".bmp"
        elif args.dataset_type == "nyu":
            gt_name = name + ".png"
        gt_file = os.path.join(gt_folder, gt_name)
        gt = np.array(Image.open(gt_file))
        # gt = cv2.resize(gt, dsize=(512, 512))
        # import pdb; pdb.set_trace()
        """for RESC"""
        if args.dataset_type == "resc":
            gt = convert_resc_labels(gt)
        elif args.dataset_type == "duke":
            gt = convert_duke_labels(gt)
        elif args.dataset_type == "nyu":
            gt = convert_our_dataset_labels(gt)

        gt_list.append(gt)
    return pred_list, gt_list


def get_hausdorff_distance(pred_list, gt_list):
    from hausdorff import hausdorff_distance

    res = {}
    weighted_total_count = 0
    weighted_total_dis = 0
    for target_class, c in enumerate(categories):
        if c == "background":
            continue
        total_dis = 0
        total_num = 0
        for i in range(len(pred_list)):
            if target_class not in gt_list[i]:
                continue
            total_num += 1
            target_class_pred = (pred_list[i] == target_class) * 1
            target_class_gt = (gt_list[i] == target_class) * 1
            total_dis += hausdorff_distance(target_class_pred, target_class_gt)
        res[c] = total_dis / total_num
        weighted_total_count += total_num
        weighted_total_dis += total_dis
    res["avg"] = sum(res.values()) / len(res)
    res["weighted_avg"] = weighted_total_dis / weighted_total_count
    return res


if __name__ == "__main__":
    # python metrics.py  --type npy --curve True
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict_dir",
        default="/scr2/xhu/jiaqi/wsss/anomaly-guided/outputs/resc_512/tieqiao_resc/cams",
        type=str,
    )
    parser.add_argument(
        "--gt_dir",
        default="/scr2/xhu/jiaqi/wsss/anomaly-guided/datasets/RESC/valid/label_images",
        type=str,
    )
    parser.add_argument(
        "--dataset_type", default="resc", type=str, choices=["duke", "resc", "nyu"]
    )

    parser.add_argument("--type", default="npy", choices=["npy", "png"], type=str)
    parser.add_argument("--t", default=0.72, type=float)
    parser.add_argument("--curve", default=True, type=bool)
    # parser.add_argument('--log_dir', default='seam_duke', type=str)
    args = parser.parse_args()

    if args.type == "npy":
        assert args.t is not None or args.curve
    if args.dataset_type == "resc":
        categories = ["background", "SRF", "PED"]
    elif args.dataset_type == "duke":
        categories = ["background", "Fluid"]
    elif args.dataset_type == "nyu":
        categories = ["background", "SRF", "IRF", "EZ", "HRD"]

    # df = pd.read_csv(args.list, names=['filename'])
    # name_list = [pth.split('/')[-1].split('.')[0] for pth in glob.glob(args.predict_dir + "/*")] #df['filename'].values
    name_list = [
        pth.split("/")[-1].replace(".npy", "")
        for pth in glob.glob(args.predict_dir + "/*")
    ]
    # import pdb; pdb.set_trace()
    print("Disease size: ", len(name_list))
    if not args.curve:
        pred_list, gt_list = do_python_eval(
            args.predict_dir, args.gt_dir, name_list, args.type, args.t
        )
        score = scores(gt_list, pred_list, n_class=len(categories))
        print(score)
        print(Dice(gt_list, pred_list, n_class=len(categories)))
        print(get_hausdorff_distance(pred_list, gt_list))
    else:
        l = []
        thres = []
        for i in range(0, 101, 1):
            t = i / 100.0
            pred_list, gt_list = do_python_eval(
                args.predict_dir, args.gt_dir, name_list, args.type, t
            )
            # import pdb; pdb.set_trace()
            score = scores(gt_list, pred_list, n_class=len(categories))
            print(score)
            l.append(
                {
                    "BG": score["Class IoU"][0],
                    "SRF": score["Class IoU"][1],
                    "PED": score["Class IoU"][2],
                    "mIoU": score["Mean IoU"],
                }
            )
            # l.append({'BG': score["Class IoU"][0], 'Fluid': score["Class IoU"][1], 'mIoU': score['Mean IoU']})
            # l.append({'BG': score["Class IoU"][0], 'SRF': score["Class IoU"][1], 'IRF': score["Class IoU"][2], 'EZ': score["Class IoU"][3], 'HRD': score["Class IoU"][4], 'mIoU': score['Mean IoU']})
            print(
                "%d/60 background score: %.3f\tmIoU: %.3f%% \n"
                % (i, t, score["Mean IoU"])
            )

            dice_score = Dice(gt_list, pred_list, n_class=len(categories))
            print(dice_score)
            thres.append(
                {
                    "threshold": t,
                    "baseline": "DFP",
                    "dataset": "RESC",
                    "SRF": score["Class IoU"][1],
                    "PED": score["Class IoU"][2],
                    "Background": score["Class IoU"][0],
                    "miou": score["Mean IoU"],
                }
            )
            print(
                "%d/60 background score: %.3f\tmIoU: %.3f%%" % (i, t, score["Mean IoU"])
            )
            print(
                "%d/60 background score: %.3f\tmDice: %.3f%%"
                % (i, t, dice_score["Macro Mean Dice"])
            )
        thres = pd.DataFrame(thres)
        thres.to_csv(
            "/scr2/xhu/jiaqi/wsss/baseline_models/supplementary/dfp_resc_scores.csv"
        )
        import pdb

        pdb.set_trace()
