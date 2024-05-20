"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function
from operator import mod

from options import Options
from lib.data import load_data
from lib.model import Ganomaly
import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

##
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
def inference():
    """ inference """

    ##
    # ARGUMENTS
    opt = Options().parse()
    ##
    
    ##
    # override by inference for ARGUMENTS
    opt.batchsize = 1
    opt.phase = 'test'
    opt.save_test_images = True
    opt.isTrain = False
    # Evaluation metric. auprc | roc | f1_score
    opt.metric = 'roc'
    opt.dataroot = '../../datasets/oct_kaggle/'
    #
    
    # LOAD DATA
    splits = ['test']
    transform = transforms.Compose([
                                    transforms.Resize((opt.isize, opt.isize)),
                                    # transforms.CenterCrop(opt.isize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
                                    ])
    dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}
    print(dataset['test'].class_to_idx)
    
    dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                batch_size=opt.batchsize,
                                                shuffle=False,
                                                num_workers=int(opt.workers),
                                                drop_last=False,
                                                worker_init_fn=(None if opt.manualseed == -1
                                                else lambda x: np.random.seed(opt.manualseed)))
                    for x in splits}
    ##
    # LOAD MODEL
    model = Ganomaly(opt, dataloader)

    ##
    # TRAIN MODEL
    model.inference(model_path='train/weights/best')

if __name__ == '__main__':
    #python inference.py --save_test_images --load_weights
    inference()
    # image = Image.open('../our_dataset/dataset_DME/1/images/DME-15307-1.jpeg').convert('RGB')
    # print(image.size)
    # transform = transforms.Compose([transforms.Resize((256, 256)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    #                                 ])
    # image = transform(image)#.repeat(3,1,1)
    # print(image.shape)
    # # image.save('test.jpeg')
    # vutils.save_image(image, 'test.jpeg', normalize=True)

