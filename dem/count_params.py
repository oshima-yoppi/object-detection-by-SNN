#%%import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as SF
from snntorch import surrogate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import itertools
import cv2
from tqdm import tqdm
# from collections import defaultdict

from module.custom_data import LoadDataset
from module import custom_data, network, compute_loss, view
from module.const import *

import matplotlib.pyplot as plt
from IPython.display import HTML

from collections import defaultdict

net = NET
net.load_state_dict(torch.load(MODEL_PATH))
#%%
def count_neuron(net):
    network_lst = net.network_lst
    neurons = 0
    width = net.input_width
    height = net.input_height
    for models in network_lst:
        for layer in models.modules():
            if isinstance(layer, torch.nn.Conv2d):
                neurons += height* width * layer.out_channels
                print(neurons)
count_neuron(net)
#%%
# Define your network model here
model = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),
    # torch.nn.Flatten(),
)

# Count the total number of neurons in the model
total_neurons = 0
for layer in model.modules():
    print(8)
    print(layer)
    # print(layer.kernel_size[0
    if isinstance(layer, torch.nn.Linear):
        total_neurons += layer.weight.numel()
    elif isinstance(layer, torch.nn.Conv2d):
        # print(layer.kernel_size)
        # print(layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels)
        total_neurons += layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels

print(f"Total number of neurons in the model: {total_neurons}")
