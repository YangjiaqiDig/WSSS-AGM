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
def load_gan_model(path, DEVICE_NR):
    ##
    # ARGUMENTS
    opt = Options().inference_parse()

    # override by inference for ARGUMENTS
    opt.batchsize = 3
    opt.ngpu = len(DEVICE_NR.split(','))
    opt.gpu_ids = DEVICE_NR
    opt.device = 'cuda'
    # print(opt)
    # Evaluation metric. auprc | roc | f1_score
    model = Ganomaly(opt, path)
    return model

if __name__ == '__main__':
    #python inference.py --save_test_images --load_weights
    print('Done')

