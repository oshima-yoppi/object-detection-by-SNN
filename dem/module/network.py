import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as SF
from snntorch import surrogate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from tonic import DiskCachedDataset
# import tonic

import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm

from . import compute_loss
import matplotlib.pyplot as plt
from IPython.display import HTML






class BaseFunction:
    def __call__(self, data):
        soft = nn.Softmax2d()
        spk_rec = []
        utils.reset(self.network)  # resets hidden states for all LIF neurons in net
        # print(data.shape)

        for step in range(data.size(0)):  # data.size(0) = number of time steps
            spk_out, mem_out = self.network(data[step])
            # print(spk_out.shape)
            if self.reshape_bool:
                batch = len(spk_out)
                spk_out = spk_out.reshape(batch, 2, self.input_height, self.input_width)

            spk_rec.append(spk_out)

        spk_rec = torch.stack(spk_rec)
        # print(spk_rec.shape)
        spk_cnt = compute_loss.spike_count(spk_rec, channel=True)# batch channel(n_class) pixel pixel 
        spk_cnt_ = spk_cnt[0,0,:,:].reshape(self.input_height, self.input_width).to('cpu').detach().numpy().copy()
        
        # print(np.sum(spk_cnt_.reshape(-1)))
       
        pred_pro = F.softmax(spk_cnt, dim=1)
        pred_pro_ = pred_pro[0,0,:,:].reshape(self.input_height, self.input_width).to('cpu').detach().numpy().copy()
        pred_pro__ = pred_pro[0,1,:,:].reshape(self.input_height, self.input_width).to('cpu').detach().numpy().copy()
        
       
        return pred_pro

class FullyConv3(BaseFunction):
    def __init__(self, beta, spike_grad, input_channel, device, input_height, input_width,  reshape_bool = True,parm_learn=True):
        self.parm_learn = True
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        c0 = input_channel
        c1 = 16
        c2 = 32
        c3 = 64
        n_class=2
        encode_kernel = 5
        decode_kernel = 5
        n_neuron = 4096
        self.network = nn.Sequential(
                    nn.Conv2d(c0, c1, encode_kernel, padding=encode_kernel//2),
                    nn.MaxPool2d(2, stride=2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
                    

                    nn.Conv2d(c1, c2, encode_kernel, padding=encode_kernel//2),
                    nn.MaxPool2d(2, stride=2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Conv2d(c2, c3, encode_kernel, padding=encode_kernel//2),
                    nn.MaxPool2d(2, stride=2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

                    nn.Conv2d(c3, c3, encode_kernel, padding=encode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
                    
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(c3, c2, decode_kernel, padding=encode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),

                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(c2, c1, decode_kernel, padding=encode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(c1, c0, decode_kernel, padding=decode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
                    
                    
                    nn.Conv2d(c0, n_class, 1,),
                    nn.AdaptiveMaxPool2d((self.input_height, self.input_width)),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, learn_beta=parm_learn, learn_threshold=parm_learn),
                    
                    ).to(device)


class FullyConv(BaseFunction):
    def __init__(self, beta, spike_grad, device,pixel=64,  reshape_bool = True,parm_learn=True, ):
        self.parm_learn = True
        self.reshape_bool = reshape_bool
        self.pixel = pixel
        c0 = 1
        c1 = 16
        c2 = 16
        n_class=2
        encode_kernel = 5
        decode_kernel = 5
        n_neuron = 4096
        self.network = nn.Sequential(nn.Conv2d(c0, c1, encode_kernel, padding=encode_kernel//2),
                    nn.MaxPool2d(2, stride=2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
                    

                    nn.Conv2d(c1, c2, encode_kernel, padding=encode_kernel//2),
                    nn.MaxPool2d(2, stride=2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Conv2d(c2, c2, encode_kernel, padding=encode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
                    
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(c2, c1, decode_kernel, padding=encode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(c1, c0, decode_kernel, padding=decode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
                    
                    
                    nn.Conv2d(c0, n_class, 1,),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, learn_beta=parm_learn, learn_threshold=parm_learn),
                    ).to(device)



class ConvDense1(BaseFunction):
    """
    
    """
    def __init__(self, beta, spike_grad, device,pixel=64,  reshape_bool = True,parm_learn=True, ):
        self.parm_learn = True
        self.reshape_bool = reshape_bool
        self.pixel = pixel
        c0 = 1
        c1 = 16
        c2 = 16
        n_class=2
        encode_kernel = 3
        decode_kernel = 3
        n_neuron = 4096
        self.network = nn.Sequential(
            nn.Conv2d(c0, c2, encode_kernel, padding=encode_kernel//2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
            

            # nn.Conv2d(c1, c2, encode_kernel, padding=encode_kernel//2),
            # nn.MaxPool2d(2, stride=2),
            # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            
            nn.Conv2d(c2, c2, encode_kernel, padding=encode_kernel//2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
            
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(c2, c1, decode_kernel, padding=encode_kernel//2),
            # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c2, c0, decode_kernel, padding=decode_kernel//2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
            
            
            nn.Conv2d(c0, n_class, 1,),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
            
            nn.Flatten(),
            nn.Linear(pixel*pixel*2, n_neuron),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
            
            nn.Linear(n_neuron, pixel*pixel*2),
            # nn.Linear(pixel*pixel*2, pixel*pixel*2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output = True,learn_beta=parm_learn, learn_threshold=parm_learn),
        ).to(device)
    
class ConvDense0(BaseFunction):
    def __init__(self, beta, spike_grad, device,pixel=64,  reshape_bool = True,parm_learn=True, ):
        self.parm_learn = True
        self.reshape_bool = reshape_bool
        self.pixel = pixel
        c0 = 1
        c1 = 16
        c2 = 16
        n_class=2
        encode_kernel = 3
        decode_kernel = 3
        n_neuron = 4096
        self.network = nn.Sequential(
            nn.Conv2d(c0, c2, encode_kernel, padding=encode_kernel//2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
            

            # nn.Conv2d(c1, c2, encode_kernel, padding=encode_kernel//2),
            # nn.MaxPool2d(2, stride=2),
            # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            
            nn.Conv2d(c2, c2, encode_kernel, padding=encode_kernel//2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
            
            # nn.Upsample(scale_factor=2),
            # nn.Conv2d(c2, c1, decode_kernel, padding=encode_kernel//2),
            # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c2, c0, decode_kernel, padding=decode_kernel//2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
            
            
            nn.Conv2d(c0, n_class, 1,),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
            
            nn.Flatten(),
            # nn.Linear(pixel*pixel*2, n_neuron),
            # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
            
            # nn.Linear(n_neuron, pixel*pixel*2),
            nn.Linear(pixel*pixel*2, pixel*pixel*2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output = True,learn_beta=parm_learn, learn_threshold=parm_learn),
        ).to(device)
# spike_grad = surrogate.atan()
def cnn(beta, spike_grad):
    reshape_bool = False
    net = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 32, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(5408, 64*64),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
                    # nn.Linear(1024, 2),
                    # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    )
    return net, reshape_bool



def fcn2(beta, spike_grad,pixel=64, parm_learn=True):
    c0 = 1
    c1 = 16
    c2 = 16
    n_class=2
    encode_kernel = 3
    decode_kernel = 3
    reshape_bool = False
    net = nn.Sequential(nn.Conv2d(c0, c2, encode_kernel, padding=encode_kernel//2),
                    nn.MaxPool2d(2, stride=2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
                    

                    # nn.Conv2d(c1, c2, encode_kernel, padding=encode_kernel//2),
                    # nn.MaxPool2d(2, stride=2),
                    # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Conv2d(c2, c2, encode_kernel, padding=encode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
                    
                    # nn.Upsample(scale_factor=2),
                    # nn.Conv2d(c2, c1, decode_kernel, padding=encode_kernel//2),
                    # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(c2, c0, decode_kernel, padding=decode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, learn_threshold=parm_learn),
                    
                    
                    nn.Conv2d(c0, n_class, 1,),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, learn_beta=parm_learn, learn_threshold=parm_learn),
                    )
    return net, reshape_bool
def fcn1(beta, spike_grad,pixel=64):
    c0 = 1
    c1 = 16
    c2 = 32
    n_class=2
    encode_kernel = 3
    decode_kernel = 2
    reshape_bool = False
    net = nn.Sequential(nn.Conv2d(c0, c1, encode_kernel, padding=encode_kernel//2),
                    nn.MaxPool2d(2, stride=2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    

                    nn.Conv2d(c1, c2, encode_kernel, padding=encode_kernel//2),
                    nn.MaxPool2d(2, stride=2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Conv2d(c2, c2, encode_kernel, padding=encode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(c2, c1, decode_kernel, padding=encode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(c1, c0, decode_kernel, padding=encode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    
                    nn.Conv2d(c0, n_class, 1, padding=encode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
                    )
    return net, reshape_bool
