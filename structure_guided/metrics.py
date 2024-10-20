"""Evaluation Metrics for Semantic Segmentation"""

import numpy as np
import os
from utils.utils import (
    convert_duke_labels,
    convert_our_dataset_labels,
    # convert_resc_labels,
)
from utils.metrics import (
    Dice,
    scores,
    record_score,
)
from PIL import Image
import argparse
import glob
import cv2
"""mIoU"""
def convert_resc_labels(img):
    # 0 background,
    # 1 lesion(need turn to background),  1 -> 0
    # 0.74 SRF(need turn to 1),           0.74 -> 1
    # 0.51 PED(need turn to 2)            0.51 -> 2
    # back: 0, ped: 128, srf: 191, retinal: 255
    img[img == 255] = 0
    img[img == 191] = 1
    img[img == 128] = 2
    return img

# categories = [
#     "background",
#     "SRF",
#     "IRF",
#     "EZ disrupted",
#     "HRD",
# ]  # ['background', 'SRF', 'PED']   # ['background', 'Fluid'] #
categories=['background', 'SRF', 'PED']

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
            predict_dict = np.load(predict_file, allow_pickle=True)

            # normalize the cam
            # cam_max = np.max(predict_dict, (1, 2), keepdims=True)
            # cam_min = np.min(predict_dict, (1, 2), keepdims=True)
            # predict_dict = (predict_dict - cam_min) / (cam_max - cam_min + 1e-5)

            bg_score = [np.ones_like(predict_dict[0]) * threshold]
            tensor = np.concatenate((bg_score, predict_dict))

            predict = np.argmax(tensor, axis=0).astype(np.uint8)
            # if len(np.unique(predict)) == 3:
            #     import pdb; pdb.set_trace()
            # predict = cv2.resize(predict, dsize=(512, 512))
            # print(predict.shape, np.unique(predict))
        pred_list.append(predict)

        """Duke only"""
        # ours_gt_name = name.split('+')[-1] + '.png'
        ours_gt_name = '%s.bmp'%name
        # meera vs mina
        # name = name.replace("meera", "mina")  # f"{name}_mina"
        # ours_gt_name = "%s.png" % name

        gt_file = os.path.join(gt_folder, ours_gt_name)
        gt = np.array(Image.open(gt_file))
        # gt = cv2.resize(gt, dsize=(512, 512))
        # import pdb; pdb.set_trace()
        """for RESC"""
        gt = convert_resc_labels(gt)
        # gt = convert_duke_labels(gt)
        # gt = convert_our_dataset_labels(gt)

        gt_list.append(gt)
    return pred_list, gt_list


if __name__ == "__main__":
    # python metrics.py  --type npy --curve True
    parser = argparse.ArgumentParser()
    # parser.add_argument("--list", default='abl_backbone_maps/resc_mnasnet/*', type=str) #duke_cam
    # parser.add_argument("--list", default='baseline_models/WSMIS/irn/result/cam/*', type=str)
    parser.add_argument("--list", default='/data2/xiaolhu/jiaqi/baseline_models/TPRO/work_dirs/resc/pseudo_labels/*', type=str)
    # parser.add_argument(
    #     "--list",
    #     default="/data2/xiaolhu/jiaqi/baseline_models/TPRO/work_dirs/nyu/cams/*",
    #     type=str,
    # )  # duke_cam
    # parser.add_argument(
    #     "--list", default="/data2/xiaolhu/jiaqi/baseline_models/TPRO/work_dirs/resc/cams/*", type=str
    # )  # resc_cam
    # parser.add_argument(
    #     "--predict_dir",
    #     default="/data2/xiaolhu/jiaqi/baseline_models/TPRO/work_dirs/nyu/cams/",
    #     type=str,
    # )
    parser.add_argument(
        "--predict_dir", default="/data2/xiaolhu/jiaqi/baseline_models/TPRO/work_dirs/resc/pseudo_labels/", type=str
    )
    # parser.add_argument(
    #     "--gt_dir", default="../datasets/our_dataset/annot_combine", type=str
    # )
    parser.add_argument("--gt_dir", default='../datasets/RESC/train/label_images', type=str)
    # parser.add_argument("--gt_dir", default='../datasets/2015_BOE_Chiu/segment_annotation/labels', type=str)

    parser.add_argument("--type", default="npy", choices=["npy", "png"], type=str)
    # parser.add_argument('--type', default='npy', choices=['npy', 'png'], type=str)

    parser.add_argument(
        "--t", default=0.6, type=float
    )  # 0.3 is the highest mIoU by curve
    parser.add_argument("--curve", default=True, type=bool)
    # parser.add_argument("--log_dir", default="resc_train_curve", type=str)
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
        for i in range(66, 75, 1):
            t = i / 100.0
            pred_list, gt_list = do_python_eval(
                args.predict_dir, args.gt_dir, name_list, args.type, t
            )
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
