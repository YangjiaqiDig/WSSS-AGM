import torch
from tqdm import tqdm

from run_train import Train
from dataset import (DukeDataset, OCTDataset, RESCDataset)
from refine_pseudo_label import refine_input_by_cam
from utils.utils import CAMGeneratorAndSave, diff_map_for_att, get_num_classes
from utils.metrics import Dice, calculate_classification_infer, scores, record_score
import time

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
        self.num_input_channel = infer_dataset[0]['image'].shape[0] * 2 if self.args.input_gan else infer_dataset[0]['image'].shape[0]
        dataloader = torch.utils.data.DataLoader(
            infer_dataset,
            num_workers=8,
            batch_size=self.args.valid_batch_size, shuffle=False)
        
        multi_task_model, _, _ = self.get_models()
        multi_task_model.eval()
        CAMGenerationModule = CAMGeneratorAndSave(opts=self.args, multi_task_model=multi_task_model)
        tensor_for_att = None
        gt_list, cam_list = [], []
        cls_sig_pred, cl_gt = [], []
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image, labels, mask = data["image"].to(self.device), data["labels"].to(self.device), data['mask'].to(self.device)        
            updated_image = image.clone()
            with torch.no_grad():
                if self.args.input_gan:
                    gan_inputs = self.transform_norml(updated_image)
                    healthy_img = self.gan_pretrained.inference(gan_inputs)
                    if self.args.att_module:
                        tensor_for_att = diff_map_for_att(updated_image, healthy_img, mask)
                        updated_image = torch.cat((image, healthy_img, tensor_for_att), dim=1)
                    else:
                        updated_image = torch.cat((image, healthy_img), dim=1)
            '''no refinement in inference generates better result'''
            with torch.no_grad():
                cls_outputs = multi_task_model(updated_image)
                sig_prediction = self.sigmoid(cls_outputs)
                cls_sig_pred += sig_prediction[:, :-1].cpu().tolist()
                cl_gt += labels[:, :-1].cpu().tolist()

            params = {'inputs': data, 'batch_preds': sig_prediction, 'refined': updated_image}
            gt_res, pred_res = CAMGenerationModule.get_cam_and_save(params, is_inference)  
            gt_list += gt_res
            cam_list += pred_res
        print(len(cam_list))
        # import pdb; pdb.set_trace()
        res_dic, auc_dic = calculate_classification_infer(cls_sig_pred, cl_gt)
        import pdb; pdb.set_trace()
        score = scores(gt_list, cam_list, n_class=get_num_classes() + 1)
        print(score)
        record_score(score, 'resc')
        dice_score = Dice(gt_list, cam_list, n_class=get_num_classes() + 1)
        print(dice_score)
        # score_sd = 0
        # ccc = 0
        # from medpy.metric.binary import assd
        # for g, p in zip(gt_list, cam_list):
        #     try:
        #         score_sd += assd(g, p)
        #         ccc += 1
        #     except:
        #         # import pdb; pdb.set_trace()
        #         continue
        # print('assd score: ', score_sd/ccc, ccc)
        return len(dataloader)
    
    def prepare_pesudo_label_for_seg(self):
        # RESC: 3093 for train + 388 for valid = 3481
        # DUKE: 7235 for train + 78 for valid
        # NYU: 3962 for train + 99 for valid (only 99 have overlap between human)
        # import pdb; pdb.set_trace()
        self.args.continue_train = True
        if 'our_dataset' in self.args.root_dirs:
            infer_dataset = OCTDataset(self.args, data_type='train', is_generate_pseudo_label=True)
        elif 'RESC' in self.args.root_dirs:
            infer_dataset = RESCDataset(self.args, data_type='train', is_generate_pseudo_label=True)
        else:
            infer_dataset = DukeDataset(self.args, data_type='train', is_generate_pseudo_label=True)
        self.num_input_channel = infer_dataset[0]['image'].shape[0] * 2 if self.args.input_gan else infer_dataset[0]['image'].shape[0]
        dataloader = torch.utils.data.DataLoader(
            infer_dataset,
            num_workers=8,
            batch_size=self.args.valid_batch_size, shuffle=False)
        
        multi_task_model, _, _ = self.get_models()
        multi_task_model.eval()
        CAMGenerationModule = CAMGeneratorAndSave(opts=self.args, multi_task_model=multi_task_model)
        tensor_for_att = None
        gt_list, cam_list = [], []
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image, _, mask = data["image"].to(self.device), data["labels"].to(self.device), data['mask'].to(self.device)          
            updated_image = image.clone()
            with torch.no_grad():
                if self.args.input_gan:
                    gan_inputs = self.transform_norml(updated_image)
                    healthy_img = self.gan_pretrained.inference(gan_inputs)
                    if self.args.att_module:
                        tensor_for_att = diff_map_for_att(updated_image, healthy_img, mask)
                        updated_image = torch.cat((image, healthy_img, tensor_for_att), dim=1)
                    else:
                        updated_image = torch.cat((image, healthy_img), dim=1)
            # updated_image = refine_input_by_cam(self.device, multi_task_model, updated_image, mask)
            # with torch.no_grad():
            #     multi_task_model.assign_conditions(False)
            params_seg = {'inputs': data, 'refined': updated_image}
            gt_res, pred_res = CAMGenerationModule.get_pseudo_labels(params_seg)  
            gt_list += gt_res
            cam_list += pred_res
        # print(len(cam_list))
        import pdb; pdb.set_trace()
        # score = scores(gt_list, cam_list, n_class=get_num_classes() + 1)
        # print(score)
        # record_score(score, 'resc')
        return len(dataloader)
    
if __name__ == "__main__":
    is_inference=True
    validator = Inference(is_inference)
    # validator.inference(infer_list=['subject_1_10.png'])
    # validator.inference(infer_list=['sn29218_78.bmp', 'sn22698_57.bmp'])
    start = time.time()
    # num_examples = validator.prepare_pesudo_label_for_seg()
    # num_examples =validator.inference(infer_list=['sn21294_59.bmp'])
    num_examples =validator.inference()
    end = time.time() - start
    print('total time: ', end, 'Avg: ', end/num_examples)


    # validator.inference(infer_list=['DME-15307-1.jpeg',
    #                               'DME-4240465-41.jpeg', 
    #                               'DR10.jpeg',
    #                               'NORMAL-15307-1.jpeg'])
