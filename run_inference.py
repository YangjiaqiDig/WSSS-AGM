from .run_train import Train
import logging
import os

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, GuidedBackpropReLUModel
import torch
from tqdm import tqdm

CUDA_DEVICE_ORDER='PCI_BUS_ID'
DEVICE_NR = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_NR
logging.basicConfig(level=logging.DEBUG)
from dataset import (DukeDataset, OCTDataset, RESCDataset)
from refine_pseudo_label import refine_input_by_cam
from utils import diff_map_for_att
from metrics import scores, record_score

class Inference(Train):
    def __init__(self, is_inference):
        super().__init__(is_inference)
        
    def inference(self, infer_list=[]):
        # if not give infer list of image names, then default as infer all testing
        self.args.continue_train = True
        self.train_parameters()
        if 'our_dataset' in self.args.root_dirs:
            curr_dataset = OCTDataset(self.args, data_type='inference', infer_list=infer_list)
        elif 'RESC' in self.args.root_dirs:
            curr_dataset = RESCDataset(self.args, data_type='inference', infer_list=infer_list)
        elif 'BOE' in self.args.root_dirs:
            curr_dataset = DukeDataset(self.args, data_type='inference', infer_list=infer_list)

        infer_dataset = self.dataset_test if not len(infer_list) else curr_dataset
        dataloader = torch.utils.data.DataLoader(
            infer_dataset,
            num_workers=8,
            batch_size=self.args.valid_batch_size, shuffle=False)
        self.cam_model.eval()
        
        cam = GradCAM(model=self.cam_model, use_cuda=self.device, target_layers=self.target_layers)
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
                updated_image = torch.cat((image, healthy_img), dim=1)
            if self.args.num_iteration > 0:
                updated_image = refine_input_by_cam(self.args, self.cam_model, updated_image, mask, cam)
            outputs = self.cam_model(updated_image, tensor_for_att)
            sig_prediction = self.sigmoid(outputs)
            # maybe only for args.n_epochs in first condition
            params = {'inputs': data, 'batch_preds': sig_prediction, 'refined': updated_image}
            gt_res, pred_res = self.CAMGenerationModule.get_cam_and_save(params)  
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
        