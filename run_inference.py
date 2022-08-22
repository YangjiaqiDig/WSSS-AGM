from .run_train import Train
import logging
import os

from pytorch_grad_cam import GradCAM
import torch
from tqdm import tqdm

CUDA_DEVICE_ORDER='PCI_BUS_ID'
DEVICE_NR = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NR
logging.basicConfig(level=logging.DEBUG)
from dataset import (DukeDataset, OCTDataset, RESCDataset)
from refine_pseudo_label import refine_input_by_cam
from utils import CAMGeneratorAndSave, diff_map_for_att
from metrics import scores, record_score

class Inference(Train):
    def __init__(self, is_inference):
        super().__init__(is_inference)
        
    def inference(self, infer_list=[]):
        # if not give infer list of image names, then default as infer all testing
        self.args.continue_train = True
        if not infer_list:
            _, infer_dataset = self.get_dataset()
        else:
            if 'our_dataset' in self.args.root_dirs:
                infer_dataset = OCTDataset(self.args, data_type='inference', infer_list=infer_list)
            elif 'RESC' in self.args.root_dirs:
                infer_dataset = RESCDataset(self.args, data_type='inference', infer_list=infer_list)
            elif 'BOE' in self.args.root_dirs:
                infer_dataset = DukeDataset(self.args, data_type='inference', infer_list=infer_list)

        dataloader = torch.utils.data.DataLoader(
            infer_dataset,
            num_workers=8,
            batch_size=self.args.valid_batch_size, shuffle=False)
        
        multi_task_model, _, _ = self.get_models()
        multi_task_model.eval()
        target_layers = [multi_task_model.module.SharedNet.base_model[-1][-1]] # .module. if use dataparallel
        with GradCAM(model=multi_task_model, use_cuda=self.device, target_layers=target_layers) as cam:
            CAMGenerationModule = CAMGeneratorAndSave(opts=self.args, cam=cam)
        tensor_for_att = None
        gt_list, cam_list = [], []
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image, _, mask = data["image"].to(self.device), data["labels"].to(self.device), data['mask'].to(self.device)        
            updated_image = image.clone()
            if self.args.input_gan:
                gan_inputs = self.transform_norml(updated_image)
                healthy_img = self.gan_pretrained.inference(gan_inputs)
                if self.args.att_module:
                    tensor_for_att = diff_map_for_att(updated_image, healthy_img, mask)
                    updated_image = torch.cat((image, healthy_img, tensor_for_att), dim=1)
                else:
                    updated_image = torch.cat((image, healthy_img), dim=1)
            if self.args.num_iteration > 0:
                updated_image = refine_input_by_cam(self.device, multi_task_model, updated_image, mask)
                
            cls_outputs, _ = multi_task_model(updated_image)
            sig_prediction = self.sigmoid(cls_outputs)
            # maybe only for args.n_epochs in first condition
            params = {'inputs': data, 'batch_preds': sig_prediction, 'refined': updated_image}
            gt_res, pred_res = CAMGenerationModule.get_cam_and_save(params, is_inference)  
            gt_list += gt_res
            cam_list += pred_res
        print(len(cam_list))

        score = scores(gt_list, cam_list, n_class=3)
        print(score)
        record_score(score, 'resc')
        
if __name__ == "__main__":
    is_inference=True
    validator = Inference(is_inference)
    # trainer.inference(infer_list=['sn8828_75.bmp', 'sn22697_89.bmp','sn29218_71.bmp'])
    validator.inference()
    # trainer.inference(infer_list=['DME-15307-1.jpeg',
    #                               'DME-4240465-41.jpeg', 
    #                               'DR10.jpeg',
    #                               'NORMAL-15307-1.jpeg'])
        