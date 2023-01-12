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
                    nn.Conv2d(12, c2, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(5408, 64*64),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
                    # nn.Linear(1024, 2),
                    # snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    )
    return net
def fcn(beta, spike_grad):
    c0 = 1
    c1 = 16
    c2 = 32
    net = nn.Sequential(nn.Conv2d(c0, c1, 4),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Conv2d(c1, c2, 4),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Conv2d(c2, c2, 4),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(c2, c1, 4),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(c1, c0, 4),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    
                    nn.Conv2d(c0, c0, 4),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
                    )
    return net