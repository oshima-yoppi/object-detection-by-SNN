import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as SF
from snntorch import surrogate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from tonic import DiskCachedDataset
# import tonic

import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm


import matplotlib.pyplot as plt
from IPython.display import HTML





# spike_grad = surrogate.atan()
def cnn(beta, spike_grad):
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
    return net
def fcn2(beta, spike_grad,pixel=64):
    c0 = 1
    c1 = 16
    c2 = 16
    n_class=2
    encode_kernel = 3
    decode_kernel = 3
    net = nn.Sequential(nn.Conv2d(c0, c2, encode_kernel, padding=encode_kernel//2),
                    nn.MaxPool2d(2, stride=2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    

                    # nn.Conv2d(c1, c2, encode_kernel, padding=encode_kernel//2),
                    # nn.MaxPool2d(2, stride=2),
                    # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Conv2d(c2, c2, encode_kernel, padding=encode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    # nn.Upsample(scale_factor=2),
                    # nn.Conv2d(c2, c1, decode_kernel, padding=encode_kernel//2),
                    # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(c2, c0, decode_kernel, padding=decode_kernel//2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    
                    nn.Conv2d(c0, n_class, 1,),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
                    )
    return net
def fcn1(beta, spike_grad,pixel=64):
    c0 = 1
    c1 = 16
    c2 = 32
    n_class=2
    encode_kernel = 3
    decode_kernel = 2
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
    return net
