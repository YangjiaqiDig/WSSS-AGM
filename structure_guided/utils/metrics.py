"""Evaluation Metrics for Semantic Segmentation"""
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
)
from utils.utils import (
    OrgLabels,
    convert_duke_labels,
    convert_our_dataset_labels,
    convert_resc_labels,
    get_binarized_cam_pred,
    get_num_lesions,
)
from PIL import Image
import argparse
import glob

"""Classification: accuracy + f1"""


def round_vals_in_dict(dic):
    for k, v in dic.items():
        if isinstance(v, float):
            dic[k] = round(v, 4)
    return dic


def calculate_classification_metrics(outputs, labels, list_of_cats):
    predicted = np.round(outputs.cpu().detach().numpy())
    gt = labels.cpu().detach().numpy()

    acc, f1m, f1mi, class_acc = 0, 0, 0, [0] * len(list_of_cats)
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
    for i in range(len(list_of_cats)):
        res_dic[list_of_cats[i]] = class_acc[i]
    res_dic = round_vals_in_dict(res_dic)
    return res_dic


"""Classification: input probabily"""


def calculate_roc(outputs, labels, name):
    # [batch, class]
    predicted = np.array(outputs)
    gt = np.array(labels)
    print(name, gt.sum(axis=0), np.round(predicted).sum(axis=0), gt.shape)
    # drop background class as they are all 1 then eroor roc
    if OrgLabels[0] == "BackGround":
        gt = gt[:, 1:]
        predicted = predicted[:, 1:]
    roc_class = roc_auc_score(gt, predicted, average=None)
    roc_avg = roc_auc_score(gt, predicted, average="weighted")
    if get_num_lesions() == 1:
        if "BackGround" in OrgLabels:
            res_dict = {OrgLabels[0]: 1, OrgLabels[1]: roc_class}
        else:
            res_dict = {OrgLabels[0]}
        res_dict = round_vals_in_dict(res_dict)
        roc_avg = round(roc_avg, 4)
        return roc_avg, res_dict
    res_dict = {}
    for i in range(len(OrgLabels)):
        if i == 0:
            res_dict[OrgLabels[i]] = 1
        else:
            res_dict[OrgLabels[i]] = roc_class[i - 1]
    res_dict = round_vals_in_dict(res_dict)
    roc_avg = round(roc_avg, 4)
    return roc_avg, res_dict


# we use f1 macro to report the result
def calculate_classification_infer(preds_list, gt_list):
    prob_predicted = np.array(preds_list)
    sparse_predicted = np.round(prob_predicted)
    gt = np.array(gt_list)

    f1m = f1_score(gt, sparse_predicted, average="macro", zero_division=1)
    f1mi = f1_score(gt, sparse_predicted, average="micro", zero_division=1)
    acc, class_acc = 0, [0] * get_num_lesions()
    for i, pred in enumerate(sparse_predicted):
        acc += accuracy_score(gt[i], pred)
        class_acc += (gt[i] == pred) * 1
    class_acc = class_acc / len(sparse_predicted)
    res_dic = {"acc": acc / len(sparse_predicted), "f1m": f1m, "f1mi": f1mi}
    for i in range(get_num_lesions()):
        res_dic[OrgLabels[i]] = class_acc[i]
    # import pdb; pdb.set_trace()
    print(gt.sum(axis=0), sparse_predicted.sum(axis=0), gt.shape)
    # drop background class as they are all 1 then error roc
    roc_class = roc_auc_score(gt, prob_predicted, average=None)
    roc_avg = roc_auc_score(gt, prob_predicted, average="weighted")
    auc_dic = {"mAUC:": roc_avg}
    for i in range(get_num_lesions()):
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


# micro or macro
def Dice(y_target, y_pred, n_class):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    dice_score_collect = []
    macro_hist = np.zeros((n_class, n_class))
    for lt, lp in zip(y_target, y_pred):
        single_hist = _fast_hist(lt.flatten(), lp.flatten(), n_class)
        intersection = np.diag(single_hist)
        union = single_hist.sum(axis=1) + single_hist.sum(axis=0) - intersection
        # dice_score = (2. * intersection + 1e-6) / (union + intersection + 1e-6)
        dice_score = (2.0 * intersection) / (union + intersection)
        # micro problem: some class may not appear in the batch, so the dice score is 1 (not make sense)
        dice_score_collect.append(dice_score)

        macro_hist += single_hist
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


def scores(label_trues, label_preds, n_class):
    accumulate_hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        accumulate_hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    # TP / all
    acc = np.diag(accumulate_hist).sum() / accumulate_hist.sum()
    # TP / (TP + FN)
    acc_cls = np.diag(accumulate_hist) / accumulate_hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)

    intersection = np.diag(accumulate_hist)
    union = accumulate_hist.sum(axis=1) + accumulate_hist.sum(axis=0) - intersection
    iu = intersection / union
    freq = accumulate_hist.sum(axis=1) / accumulate_hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    valid = accumulate_hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    cls_iu = dict(zip(range(n_class), iu))
    # import pdb; pdb.set_trace()
    final_dict = {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": round_vals_in_dict(cls_iu),
    }
    final_dict = round_vals_in_dict(final_dict)
    return final_dict


def record_score(score, save_path):
    score_list = []
    for i in range(get_num_lesions() + 1):
        score_list.append(score["Class IoU"][i])
        aveJ = score["Mean IoU"]
    with open("iou_collections/{}_iou.txt".format(save_path), "w") as f:
        for num, cls_iou in enumerate(score_list):
            print(
                "class {:2d} {:12} IU {:.2f}".format(
                    num, OrgLabels[num], round(cls_iou, 3)
                )
            )
            f.write(
                "class {:2d} {:12} IU {:.2f}".format(
                    num, OrgLabels[num], round(cls_iou, 3)
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


def update_scores(
    gt_for_cam, gt_for_layer_cam, cam_img_list, num_class, default_thres, each, i
):
    cam_n = each["cam_n"]
    updated_cam_list = [get_binarized_cam_pred(i, cam_n, c) for c in cam_img_list]
    if "layer" in cam_n:
        s = scores(gt_for_layer_cam, updated_cam_list, n_class=2)
    else:
        s = scores(gt_for_cam, updated_cam_list, n_class=num_class)
    if s["Mean IoU"] > each["score"]:
        each["score"] = s["Mean IoU"]
        each["best_thred"] = i / 100
    if i == default_thres * 100:
        each["default_thred_score"] = s["Mean IoU"]
    return each


def get_miou_for_multilevel_preds(
    cam_img_list, gt_for_cam, gt_for_layer_cam, num_class, default_thres
):
    all_keys = [
        {"cam_n": x, "score": 0, "best_thred": 0, f"default_thred_score": 0}
        for x in cam_img_list[0].keys()
        if x not in ["gt", "img_name"]
    ]
    for i in range(5, 100, 5):
        for each in all_keys:
            each = update_scores(
                gt_for_cam,
                gt_for_layer_cam,
                cam_img_list,
                num_class,
                default_thres,
                each,
                i,
            )
    return all_keys


"""mIoU"""
categories = [
    "background",
    "SRF",
    "IRF",
    "EZ disrupted",
    "HRD",
]  # ['background', 'SRF', 'PED']   # ['background', 'Fluid'] #


def do_python_eval(
    predict_folder, gt_folder, name_list, input_type="png", threshold=1.0
):
    pred_list = []
    gt_list = []
    # input_type = 'png'
    for name in name_list:
        if input_type == "png":
            predict_file = os.path.join(predict_folder, "%s.png" % name)
            predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)

            # for 2 experts comparison
            predict = convert_our_dataset_labels(predict)
        elif input_type == "npy":
            predict_file = os.path.join(predict_folder, "%s.npy" % name)
            predict_dict = np.load(predict_file, allow_pickle=True).item()
            h, w = list(predict_dict.values())[0].shape
            tensor = np.zeros((len(categories), h, w), np.float32)
            for key in predict_dict.keys():
                tensor[key + 1] = predict_dict[
                    key
                ]  # .cpu().numpy() # we saved seam crf_cam in gpu by mistake..
            tensor[0, :, :] = threshold

            # predict_dict = np.load(predict_file, allow_pickle=True)
            # bg_score = [np.ones_like(predict_dict[0]) * threshold]
            # tensor = np.concatenate((bg_score, predict_dict))

            predict = np.argmax(tensor, axis=0).astype(np.uint8)
            # predict = cv2.resize(predict, dsize=(512, 512))
            # print(predict.shape, np.unique(predict))
        pred_list.append(predict)

        """Duke only"""
        # duke_gt_name = name.split('+')[-1] + '.png'
        # resc_gt_name = '%s.bmp'%name
        # meera vs mina
        name = name.replace("meera", "mina")  # f"{name}_mina"
        ours_gt_name = "%s.png" % name

        gt_file = os.path.join(gt_folder, ours_gt_name)
        gt = np.array(Image.open(gt_file))
        # gt = cv2.resize(gt, dsize=(512, 512))
        # import pdb; pdb.set_trace()
        """for RESC"""
        # gt = convert_resc_labels(gt)
        # gt = convert_duke_labels(gt)
        gt = convert_our_dataset_labels(gt)

        # print(gt.shape, np.unique(gt))
        gt_list.append(gt)
    return pred_list, gt_list


if __name__ == "__main__":
    # python metrics.py  --type npy --curve True
    parser = argparse.ArgumentParser()
    # parser.add_argument("--list", default='abl_backbone_maps/resc_mnasnet/*', type=str) #duke_cam
    # parser.add_argument("--list", default='baseline_models/WSMIS/irn/result/cam/*', type=str)
    # parser.add_argument("--list", default='paper/thresh/resc/*', type=str)
    parser.add_argument(
        "--list", default="datasets/our_dataset/annotation_v2/*_meera.png", type=str
    )  # duke_cam
    parser.add_argument(
        "--predict_dir", default="datasets/our_dataset/annotation_v2/", type=str
    )
    # parser.add_argument("--predict_dir", default='baseline_models/WSMIS/irn/result/cam', type=str)
    parser.add_argument(
        "--gt_dir", default="datasets/our_dataset/annotation_v2", type=str
    )
    # parser.add_argument("--gt_dir", default='datasets/RESC/valid/label_images', type=str)
    # parser.add_argument("--gt_dir", default='datasets/2015_BOE_Chiu/segment_annotation/labels', type=str)

    parser.add_argument("--type", default="png", choices=["npy", "png"], type=str)
    # parser.add_argument('--type', default='npy', choices=['npy', 'png'], type=str)

    parser.add_argument(
        "--t", default=0.6, type=float
    )  # 0.3 is the highest mIoU by curve
    parser.add_argument("--curve", default=True, type=bool)
    parser.add_argument("--log_dir", default="resc_train_curve", type=str)
    args = parser.parse_args()

    if args.type == "npy":
        assert args.t is not None or args.curve
    # df = pd.read_csv(args.list, names=['filename'])
    name_list = [
        pth.split("/")[-1].split(".")[0] for pth in glob.glob(args.list)
    ]  # df['filename'].values
    # import pdb; pdb.set_trace()
    print("Disease size: ", len(name_list))
    if not args.curve:
        pred_list, gt_list = do_python_eval(
            args.predict_dir, args.gt_dir, name_list, args.type, args.t
        )
        score = scores(gt_list, pred_list, n_class=len(categories))
        print(score)
        # record_score(score, args.log_dir)
    else:
        l = []
        thres = []
        for i in range(60, 61, 1):
            t = i / 100.0
            pred_list, gt_list = do_python_eval(
                args.predict_dir, args.gt_dir, name_list, args.type, t
            )
            # import pdb; pdb.set_trace()
            # score_sd = 0
            # ccc = 0
            # for g, p in zip(gt_list, pred_list):
            #     try:
            #         score_sd += assd_score(g, p, n_class=len(categories))
            #         ccc += 1
            #     except:
            #         # import pdb; pdb.set_trace()
            #         continue
            # print('assd score: ', score_sd/ccc, ccc)

            score = scores(gt_list, pred_list, n_class=len(categories))
            print(score)
            dice_score = Dice(pred_list, gt_list, n_class=len(categories))
            print(dice_score)
            # record_score(score, args.log_dir)
            l.append(score["Mean IoU"])
            # thres.append({
            #     'threshold': t,
            #     'baseline': 'SEAM',
            #     'dataset': 'RESC',
            #     'SRF': score['Class IoU'][1],
            #     'PED': score['Class IoU'][2],
            #     'Background': score['Class IoU'][0],
            #     'miou': score['Mean IoU']
            # })
            print(
                "%d/60 background score: %.3f\tmIoU: %.3f%%" % (i, t, score["Mean IoU"])
            )
            print(
                "%d/60 background score: %.3f\tmDice: %.3f%%"
                % (i, t, dice_score["Macro Mean Dice"])
            )
        # thres=pd.DataFrame(thres)
        # thres.to_csv('paper/thresh/resc/seam_scores.csv')
