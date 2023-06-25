"""GANomaly
"""
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import numpy as np

import torch.utils.data
from gan_and_str.ganomaly.lib.networks import NetG, NetD, weights_init


class BaseModel():
    """ Base Model for ganomaly
    """
    def __init__(self, opt):
        ##
        # Seed for deterministic behavior
        self.seed(opt.manualseed)

        # Initalize variables.
        self.opt = opt

    ##
    def seed(self, seed_value):
        """ Seed 
        
        Arguments:
            seed_value {int} -- [description]
        """
        # Check if seed is default value
        if seed_value == -1:
            return

        # Otherwise seed all functionality
        import random
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deterministic = True
        
    def inference(self, image_tensor):
        with torch.no_grad():
            # self.netg.eval()
            # print(image_tensor.is_cuda)
            self.input.resize_(image_tensor.size()).copy_(image_tensor)
            self.fake, latent_i, latent_o = self.netg(self.input)
            return self.fake.data

            
##
# origin image + 
class Ganomaly(BaseModel):
    def __init__(self, opt, path):
        super(Ganomaly, self).__init__(opt)
        # cpu
        self.device = torch.device(self.opt.device)
        self.netg = NetG(self.opt).to(self.device)
        self.netg.apply(weights_init)
        
        with torch.no_grad():
            self.netg.eval()
            try:
                pretrained_dict = torch.load(path)['state_dict'] #, map_location='cpu'
                self.netg.load_state_dict(pretrained_dict)
            except IOError:
                raise IOError("netG weights not found")

        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
