import os
import pandas as pd
import torch
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE_NR = "5"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_NR

from utils.utils_gradcam import CAMGeneratorAndSave


from utils.utils import get_gt_and_relevant_cams, normalized_batch_tensor
from utils.metrics import get_miou_for_multilevel_preds, scores
import numpy as np
from run_train import Train

import warnings

warnings.filterwarnings("ignore")


class Inference_M(Train):
    def __init__(self):
        super().__init__(is_inference=True)

    def pred(self):
        _, dataset_test = self.get_dataset()
        testloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=8,
            batch_size=self.args.valid_batch_size,
            shuffle=False,
        )
        multi_task_model, _, _ = self.get_models()
        # import pdb; pdb.set_trace()

        multi_task_model.eval()
        # CamModel = CAMGeneratorAndSave(self.args, multi_task_model)
        cam_img_list, gradcam_img_list = (
            [],
            [],
        )
        for batch, data in tqdm(enumerate(testloader), total=len(testloader)):
            (
                image,
                labels,
                binary_labels,
                layer_img,
                layer_uncertain,
                caption_tensor,
                anomaly_diff,
            ) = (
                data["image"].to(self.device),
                data["labels"].to(self.device),
                data["binary_labels"].to(self.device),
                data[self.layer_const].to(self.device),
                data["layer_uncertain"].to(self.device),
                data["caption_tensor"].to(self.device),
                data["anomaly_diff"].to(self.device),
            )
            if self.args.add_abnormal:
                layer_img = normalized_batch_tensor(layer_img + anomaly_diff)
            if 1 not in binary_labels:
                continue
            with torch.no_grad():
                comb_input = torch.cat([image, layer_img, layer_uncertain], dim=1)
                (cls_pred_dicts, cams_dict, constraint_dict) = multi_task_model(
                    comb_input, caption_input=caption_tensor
                )

                # layer_cams = normalize_cam(cams["layer_cam"])
                updated_cams_dicts = get_gt_and_relevant_cams(
                    cams_dict,
                    data,
                    self.args,
                )

            # _, ready_gradcam_pred = CamModel.get_cam_and_save(
            #     {
            #         "inputs": data,
            #         "input_tensor": comb_input,
            #     }
            # )

            # gradcam_img_list += ready_gradcam_pred
            cam_img_list += updated_cams_dicts
            # import pdb; pdb.set_trace()

        print(f"Num of CAM images: {len(cam_img_list)}")

        gt_for_cam = [x["gt"] for x in cam_img_list]
        gt_for_layer_cam = [(x["gt"] > 0) * 1 for x in cam_img_list]
        # score_gradcam = scores(gt_for_cam, gradcam_img_list, n_class=self.num_class)
        # print("GradCAM:\n", score_gradcam)

        all_keys = get_miou_for_multilevel_preds(
            cam_img_list,
            gt_for_cam,
            gt_for_layer_cam,
            self.full_num_class,
            self.args.out_cam_pred_alpha,
        )
        # print in table format
        # import pdb; pdb.set_trace()
        print(pd.DataFrame(all_keys))


if __name__ == "__main__":
    infer = Inference_M()
    infer.pred()
