"""
INFERENCE GANOMALY

. Example: Run the following command from the terminal.
    run gan_inference.py                             \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""
from __future__ import print_function

from gan_and_str.ganomaly.options import Options
from gan_model import Ganomaly

##
def load_gan_model(pretrained_dict, device):
    ##
    # ARGUMENTS
    opt = Options().inference_parse()
    # override by inference for ARGUMENTS
    opt.batchsize = 1
    opt.ngpu = 0
    # Evaluation metric. auprc | roc | f1_score
    model = Ganomaly(opt, pretrained_dict, device)
    return model

if __name__ == '__main__':
    #python inference.py --save_test_images --load_weights
    print('Done')

