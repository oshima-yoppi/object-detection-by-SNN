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
from . import const

import matplotlib.pyplot as plt
from IPython.display import HTML


class BaseFunction(nn.Module):
    def __call__(self, data, time):
        self.spike_count = 0
        spk_rec = []
        # utils.reset(self.network)  # resets hidden states for all LIF neurons in net
        for net in self.network_lst:
            utils.reset(net)

        for step in range(time):  # data.size(0) = number of time steps
            for i, net_ in enumerate(self.network_lst):
                if i == 0:
                    data_ = net_(data[step])
                elif i < len(self.network_lst) - 1:
                    data_ = net_(data_)
                elif i == len(self.network_lst) - 1:
                    data_, mem = net_(data_)
                    # print(_.shape)
                    # print(_)
                # print(data_.shape)

                if self.power:
                    self.spike_count += torch.sum(data_)
            # spk_rec.append(data_)
        # spk_rec = torch.stack(spk_rec)
        # spk_cnt = compute_loss.spike_count(
        #     spk_rec, channel=True
        # )  # batch channel(n_class) pixel pixel
       
        # pred_pro = F.softmax(spk_cnt, dim=1)
        # pred_pro = torch.sigmoid(spk_cnt)
        pred_pro = torch.sigmoid(mem-0.5)

        # print(pred_pro.shape)
        pred_pro_ = 1 - pred_pro
        pred_pro = torch.cat([pred_pro_, pred_pro], dim=1)
        # print(mem.item())
        # print(mem)
        return pred_pro
    def count_neurons(self):
        """
        ネットワーク内のニューロンの数を数える。発火率を算出する際に使用。
        torchライブラリじゃだめかもしれないから自作
        """
        self.number_neurons = 0
        input_dummy = torch.zeros(1, self.input_channel, self.input_height, self.input_width).to(self.device)
        for i, net in enumerate(self.network_lst):
            if i == len(self.network_lst) - 1:
                input_dummy, _ = net(input_dummy)
            else:
                input_dummy = net(input_dummy)
            if input_dummy.dim() == 4:
                _, c, h, w = input_dummy.shape
                self.number_neurons += c * h * w
            elif input_dummy.dim() == 2:
                _, c = input_dummy.shape
                self.number_neurons += c
        return self.number_neurons

class SpatialAttentionBlock(nn.Module):
    def __init__(
        self,
        beta,
        spike_grad,
        input_channel,
        middle_channel,
        device,
        input_height,
        input_width,
        reshape_bool=True,
        parm_learn=True,
        reset="subtract",
        power=False,
    ):
        self.parm_learn = parm_learn
        self.device = device
        self.input_channel = input_channel
        self.middle_channel = middle_channel
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        self.power = power
        super().__init__()
        self.branched_net = nn.Sequential(
            nn.Conv2d(self.input_channel, self.middle_channel, 1),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Conv2d(self.middle_channel, 1, 1),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            ).to(device)
    def forward(self, x):
        branched_x = self.branched_net(x)
        x = x * branched_x
        return x
class ChannelAttentionBlock(nn.Module): # SeNet
    def __init__(
        self,
        beta,
        spike_grad,
        input_channel,
        middle_channel,
        device,
        input_height,
        input_width,
        reshape_bool=True,
        parm_learn=True,
        reset="subtract",
        power=False,
    ):
        self.parm_learn = parm_learn
        self.device = device
        self.input_channel = input_channel
        self.middle_channel = middle_channel
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        self.power = power
        super().__init__()
        self.branched_net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.input_channel, self.middle_channel, 1),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Conv2d(self.middle_channel, self.input_channel, 1),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ), ).to(device)
        
    def forward(self, x):
        branched_x = self.branched_net(x)
        # print(torch.sum(x), torch.sum(branched_x))
        x = x * branched_x
        # print(torch.sum(x).item())
        return x
   
class AttentionNetwork(BaseFunction):
    def __init__(
        self,
        beta,
        spike_grad,
        input_channel,
        device,
        input_height,
        input_width,
        reshape_bool=True,
        parm_learn=True,
        reset="subtract",
        power=False,
    ):
        self.parm_learn = parm_learn
        self.device = device
        self.input_channel = input_channel
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        self.power = power
        c0 = input_channel
        c1 = 64
        c2 = 128
        c3 = 512
        n1 = c3
        n2 = 128
        ratio_drop = 0.4
        encode_kernel = 3
        n_output = 1

        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(c0, c1, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.down2 = nn.Sequential(
            nn.Conv2d(c1, c2, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.down3 = nn.Sequential(
            nn.Conv2d(c2, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.attention1 = nn.Sequential(
            ChannelAttentionBlock(
                beta=beta,
                spike_grad=spike_grad,
                input_channel=c3,
                middle_channel=c1,
                device=device,
                input_height=input_height,
                input_width=input_width,
                reshape_bool=reshape_bool,
                parm_learn=parm_learn,
                reset=reset,
                power=power,
            ),
            # nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.attention2 = nn.Sequential(
            SpatialAttentionBlock(
                beta=beta,
                spike_grad=spike_grad,
                input_channel=c3,
                middle_channel=c1,
                device=device,
                input_height=input_height,
                input_width=input_width,
                reshape_bool=reshape_bool,
                parm_learn=parm_learn,
                reset=reset,
                power=power,
            ),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.lenear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n1, n2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout(ratio_drop),
        ).to(device)
        self.output = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(n2, n_output),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                output=True,
                learn_threshold=parm_learn,
                reset_mechanism='none',
            ),
        ).to(device)


        self.network_lst = [
            self.down1,
            self.down2,
            self.down3,
            self.attention1,
            self.attention2,
            self.lenear1,
            self.output
        ]
        
class Conv3GloabalFull2(BaseFunction):
    def __init__(
        self,
        beta,
        spike_grad,
        input_channel,
        device,
        input_height,
        input_width,
        reshape_bool=True,
        parm_learn=True,
        reset="subtract",
        power=False,
    ):
        self.parm_learn = parm_learn
        self.device = device
        self.input_channel = input_channel
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        self.power = power
        c0 = input_channel
        c1 = 16
        c2 = 64
        c3 = 512
        n_class = 2
        neu = 88064
        n1 = 512
        n2 = 128
        n3 = 1
        n4 = 1

        ratio_drop = 0.4

        encode_kernel = 5
        decode_kernel = 5
        # n_neuron = 4096
        n_output = 1

        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(c0, c2, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Dropout2d(ratio_drop),
            
        ).to(device)
        self.lenear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n1, n2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout(ratio_drop),
        ).to(device)
        self.output = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(n2, n_output),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                output=True,
                learn_threshold=parm_learn,
                reset_mechanism='none',
            ),
        ).to(device)
        self.network_lst = [
            self.down1,
            self.down2,
            self.down3,
            self.lenear1,
            self.output
        ]

   



class Conv2Full3_Drop(BaseFunction):
    def __init__(
        self,
        beta,
        spike_grad,
        input_channel,
        device,
        input_height,
        input_width,
        reshape_bool=True,
        parm_learn=True,
        reset="subtract",
        power=False,
    ):
        self.parm_learn = parm_learn
        self.device = device
        self.input_channel = input_channel
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        self.power = power
        c0 = input_channel
        c1 = 16
        c2 = 32
        c3 = 64
        n_class = 2
        neu = 88064
        n1 = 8960
        n2 = 4096
        n3 = 512
        n4 = 1

        ratio_drop = 0.4

        encode_kernel = 5
        decode_kernel = 5
        n_neuron = 4096
        n_output = 2

        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(c0, c2, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        # self.down3 = nn.Sequential(
        #     nn.Conv2d(c3, c3, encode_kernel, padding=encode_kernel // 2),
        #     nn.MaxPool2d(2, stride=2),
        #     snn.Leaky(
        #         beta=beta,
        #         spike_grad=spike_grad,
        #         init_hidden=True,
        #         learn_beta=parm_learn,
        #         learn_threshold=parm_learn,
        #         reset_mechanism=reset,
        #     ),
        #     nn.Dropout2d(ratio_drop),
        # ).to(device)
        self.lenear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n1, n2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout(ratio_drop),
        ).to(device)
        self.lenear2 = nn.Sequential(
            nn.Linear(n2, n3),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout(ratio_drop),
        ).to(device)
        self.lenear3 = nn.Sequential(
            nn.Linear(n3, n4),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                output=True,
                learn_threshold=parm_learn,
                reset_mechanism='none',
            ),
        ).to(device)
        self.network_lst = [
            self.down1,
            self.down2,
            self.lenear1,
            self.lenear2,
            self.lenear3,
        ]

class Conv3Full3_Drop(BaseFunction):
    def __init__(
        self,
        beta,
        spike_grad,
        input_channel,
        device,
        input_height,
        input_width,
        reshape_bool=True,
        parm_learn=True,
        reset="subtract",
        power=False,
    ):
        self.parm_learn = parm_learn
        self.device = device
        self.input_channel = input_channel
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        self.power = power
        c0 = input_channel
        c1 = 16
        c2 = 32
        c3 = 64
        n_class = 2
        neu = 88064
        n1 = 2240
        n2 = 512
        n3 = 256
        n4 = 1

        ratio_drop = 0.4

        encode_kernel = 5
        decode_kernel = 5
        n_neuron = 4096
        n_output = 2

        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(c0, c2, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.lenear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n1, n2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout(ratio_drop),
        ).to(device)
        self.lenear2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n2, n3),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout(ratio_drop),
        ).to(device)
        self.lenear3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n3, n4),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                output=True,
                learn_threshold=parm_learn,
                reset_mechanism='none',
            ),
        ).to(device)
        self.network_lst = [
            self.down1,
            self.down2,
            self.down3,
            self.lenear1,
            self.lenear2,
            self.lenear3,
        ]

   



class Conv3Full1_Drop(BaseFunction):
    def __init__(
        self,
        beta,
        spike_grad,
        input_channel,
        device,
        input_height,
        input_width,
        reshape_bool=True,
        parm_learn=True,
        reset="subtract",
        power=False,
    ):
        self.parm_learn = parm_learn
        self.device = device
        self.input_channel = input_channel
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        self.power = power
        c0 = input_channel
        c1 = 16
        c2 = 32
        c3 = 64
        n_class = 2
        neu = 88064
        n1 = 2240
        n2 = 512
        n3 = 1

        ratio_drop = 0.4

        encode_kernel = 5
        decode_kernel = 5
        n_neuron = 4096
        n_output = 2

        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(c0, c2, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout2d(ratio_drop),
        ).to(device)
        self.lenear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n1, n2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
            nn.Dropout(ratio_drop),
        ).to(device)
        self.lenear2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n1, n3),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
                output=True,
            ),
            # nn.Dropout(ratio_drop),
        ).to(device)
        self.network_lst = [
            self.down1,
            self.down2,
            self.down3,
            # self.lenear1,
            self.lenear2,
            # self.lenear3,
        ]

   


class Conv3Full3(nn.Module):
    def __init__(
        self,
        beta,
        spike_grad,
        input_channel,
        device,
        input_height,
        input_width,
        reshape_bool=True,
        parm_learn=True,
        reset="subtract",
        power=False,
    ):
        self.parm_learn = parm_learn
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        self.power = power
        c0 = input_channel
        c1 = 16
        c2 = 32
        c3 = 64
        n_class = 2
        neu = 88064
        n1 = 21504
        n2 = 4096
        n3 = 512
        n4 = 2

        encode_kernel = 5
        decode_kernel = 5
        n_neuron = 4096
        n_output = 2

        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(c0, c2, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        self.lenear1 = nn.Sequential(
            nn.Linear(n1, n2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        self.lenear2 = nn.Sequential(
            nn.Linear(n2, n3),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        self.lenear3 = nn.Sequential(
            nn.Linear(n3, n4),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                output=True,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        self.network_lst = [
            self.down1,
            self.down2,
            self.lenear1,
            self.lenear2,
            self.lenear3,
        ]

    def forward(self, data, time):
        self.spike_count = 0
        spk_rec = []
        for net in self.network_lst:
            utils.reset(net)
        for step in range(time):
            d1 = self.down1(data[step])
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d3 = d3.reshape(d3.shape[0], -1)
            l1 = self.lenear1(d3)
            l2 = self.lenear2(l1)
            l3, _ = self.lenear3(l2)
            # output, _ = self.output(u2)
            spk_rec.append(l3)
            if self.power:
                self.spike_count += torch.sum(d1)
                self.spike_count += torch.sum(d2)
                self.spike_count += torch.sum(d3)
                self.spike_count += torch.sum(l1)
                self.spike_count += torch.sum(l2)
                self.spike_count += torch.sum(l3)

        # print(self.spike_count)
        spk_rec = torch.stack(spk_rec)
        # print(spk_rec.shape)
        spk_cnt = compute_loss.spike_count(
            spk_rec, channel=True
        )  # batch channel(n_class) pixel pixel

        # print(np.sum(spk_cnt_.reshape(-1)))

        pred_pro = F.softmax(spk_cnt, dim=1)
        # print(pred_pro[0])
        # pred_pro = F.sigmoid(spk_cnt)
        return pred_pro


class Conv3Full2(nn.Module):
    def __init__(
        self,
        beta,
        spike_grad,
        input_channel,
        device,
        input_height,
        input_width,
        reshape_bool=True,
        parm_learn=True,
        reset="subtract",
        power=False,
    ):
        self.parm_learn = parm_learn
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        self.power = power
        c0 = input_channel
        c1 = 16
        c2 = 32
        c3 = 64
        n_class = 2
        neu = 88064
        n1 = 88064
        n2 = 4096
        n3 = 2

        encode_kernel = 5
        decode_kernel = 5
        n_neuron = 4096
        n_output = 2

        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(c0, c2, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        self.down2 = nn.Sequential(
            nn.Conv2d(c2, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        self.down3 = nn.Sequential(
            nn.Conv2d(c3, c3, encode_kernel, padding=encode_kernel // 2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        self.lenear1 = nn.Sequential(
            nn.Linear(n1, n2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        self.lenear2 = nn.Sequential(
            nn.Linear(n2, n3),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                output=True,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        # self.lenear3 = nn.Sequential(
        #             nn.Linear(n_neuron, n_output),
        #             snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, learn_beta=parm_learn, output = True, learn_threshold=parm_learn, reset_mechanism=reset),
        # ).to(device)
        self.network_lst = [self.down1, self.down2, self.lenear1, self.lenear2]

    def forward(self, data, time):
        self.spike_count = 0
        spk_rec = []
        for net in self.network_lst:
            utils.reset(net)
        for step in range(time):
            d1 = self.down1(data[step])
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d3 = d3.reshape(d3.shape[0], -1)
            l1 = self.lenear1(d3)
            l2, _ = self.lenear2(l1)
            # output, _ = self.output(u2)
            spk_rec.append(l2)
            if self.power:
                self.spike_count += torch.sum(d1)
                self.spike_count += torch.sum(d2)
                self.spike_count += torch.sum(d3)
                self.spike_count += torch.sum(l1)
                self.spike_count += torch.sum(l2)

        # print(self.spike_count)
        spk_rec = torch.stack(spk_rec)
        # print(spk_rec.shape)
        spk_cnt = compute_loss.spike_count(
            spk_rec, channel=True
        )  # batch channel(n_class) pixel pixel

        # print(np.sum(spk_cnt_.reshape(-1)))

        pred_pro = F.softmax(spk_cnt, dim=1)
        # print(pred_pro[0])
        # pred_pro = F.sigmoid(spk_cnt)
        return pred_pro


class FullyConv2(BaseFunction):
    def __init__(
        self,
        beta,
        spike_grad,
        input_channel,
        device,
        input_height,
        input_width,
        reshape_bool=True,
        parm_learn=True,
    ):
        self.parm_learn = parm_learn
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        c0 = input_channel
        c1 = 16
        c2 = 32
        c3 = 64
        n_class = 2
        encode_kernel = 5
        decode_kernel = 5
        n_neuron = 4096
        self.network = nn.Sequential(
            nn.Conv2d(c0, c2, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
            nn.Conv2d(c2, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
            nn.Conv2d(c3, c3, encode_kernel, padding=encode_kernel // 2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c3, c2, decode_kernel, padding=encode_kernel // 2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c2, c0, decode_kernel, padding=decode_kernel // 2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
            nn.Conv2d(
                c0,
                n_class,
                1,
            ),
            nn.AdaptiveMaxPool2d((self.input_height, self.input_width)),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                output=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
        ).to(device)


class FullyConv3(BaseFunction):
    def __init__(
        self,
        beta,
        spike_grad,
        input_channel,
        device,
        input_height,
        input_width,
        reshape_bool=True,
        parm_learn=True,
    ):
        self.parm_learn = parm_learn
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        c0 = input_channel
        c1 = 16
        c2 = 32
        c3 = 64
        n_class = 2
        encode_kernel = 5
        decode_kernel = 5
        n_neuron = 4096
        self.network = nn.Sequential(
            nn.Conv2d(c0, c1, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
            nn.Conv2d(c1, c2, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
            nn.Conv2d(c2, c3, encode_kernel, padding=encode_kernel // 2),
            nn.MaxPool2d(2, stride=2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
            nn.Conv2d(c3, c3, encode_kernel, padding=encode_kernel // 2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c3, c2, decode_kernel, padding=encode_kernel // 2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c2, c1, decode_kernel, padding=encode_kernel // 2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c1, c0, decode_kernel, padding=decode_kernel // 2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
            nn.Conv2d(
                c0,
                n_class,
                1,
            ),
            nn.AdaptiveMaxPool2d((self.input_height, self.input_width)),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                output=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
            ),
        ).to(device)


class AnnConv2(nn.Module):
    def __init__(self, input_channel, input_height, input_width, device):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        c0 = input_channel
        c1 = 16
        c2 = 32
        c3 = 64
        n_class = 2
        encode_kernel = 5
        decode_kernel = 5
        n_neuron = 4096
        self.network = nn.Sequential(
            nn.Conv2d(c0, c2, encode_kernel, padding=encode_kernel // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(c2, c3, encode_kernel, padding=encode_kernel // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(c3, c3, encode_kernel, padding=encode_kernel // 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c3, c2, decode_kernel, padding=encode_kernel // 2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c2, c1, decode_kernel, padding=decode_kernel // 2),
            nn.ReLU(),
            nn.Conv2d(
                c1,
                n_class,
                1,
            ),
            nn.AdaptiveMaxPool2d((self.input_height, self.input_width)),
        ).to(device)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        output = self.network(x)
        # pred_pro = F.softmax(output, dim=1)
        pred_pro = self.soft(output)
        # print(pred_pro.shape)
        # aaa = pred_pro.to('cpu').detach().numpy().copy()
        # plt.figure()
        # plt.imshow(aaa[0,0])
        # plt.show()
        # print(output.shape)
        # image = output[0]
        # image = image.to('cpu').detach().numpy().copy()
        # fig  = plt.figure()
        # ax1 = fig.add_subplot(121)
        # ax2  = fig.add_subplot(122)
        # ax1.imshow(image[0])
        # ax2.imshow(image[1])
        # plt.show()
        return pred_pro
