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
from . import rand

import matplotlib.pyplot as plt
from IPython.display import HTML

rand.give_seed()


class BaseFunction(nn.Module):
    def __call__(self, data, label, time, loss_func=None):
        self.spike_count = 0
        spk_rec = []
        # utils.reset(self.network)  # resets hidden states for all LIF neurons in net
        loss = 0
        for net in self.network_lst:
            utils.reset(net)
        for step in range(time):  # data.size(0) = number of time steps
            for i, net_ in enumerate(self.network_lst):
                if i == 0:
                    if self.repeat_input:
                        data_ = net_(data[0])
                    else:
                        data_ = net_(data[step])
                elif i < len(self.network_lst) - 1:
                    data_ = net_(data_)
                elif i == len(self.network_lst) - 1:
                    data_, mem = net_(data_)
                # print(data_.shape)

                if self.power:
                    self.spike_count += torch.sum(data_)
            # spk_rec.append(data_)

            if self.time_aware_loss:
                # spk_rec = torch.stack(spk_rec)
                pred_pro = torch.sigmoid(mem - 0.5)
                if loss_func is not None:
                    loss += loss_func(pred_pro, label) / time
        if self.time_aware_loss == False:
            pred_pro = torch.sigmoid(mem - 0.5)
            if loss_func is not None:
                loss += loss_func(pred_pro, label)
        # spk_rec = torch.stack(spk_rec)
        # print(spk_cnt.shape)
        # print(spk_cnt_resize.shape)
        # pred_pro = torch.sigmoid(mem - 0.5)
        batch_size = pred_pro.shape[0]
        loss /= batch_size
        return pred_pro, loss

    def count_neurons(self):
        """
        ネットワーク内のニューロンの数を数える。発火率を算出する際に使用。
        torchライブラリじゃだめかもしれないから自作
        """
        for net in self.network_lst:
            utils.reset(net)
        self.number_neurons = 0
        input_dummy = torch.zeros(
            1, self.input_channel, self.input_height, self.input_width
        ).to(self.device)
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

    def get_threshold(self):
        """
        ネットワークの閾値を調べる
        """
        self.threshold_lst = []
        for net in self.network_lst:
            utils.reset(net)
        input_dummy = torch.zeros(
            1, self.input_channel, self.input_height, self.input_width
        ).to(self.device)
        for i, net in enumerate(self.network_lst):
            for layer in net:
                if isinstance(layer, snn.Leaky):
                    self.threshold_lst.append(layer.threshold.item())
        # print(self.threshold_lst)
        return self.threshold_lst


class RoughConv3(BaseFunction):
    def __init__(
        self,
        beta,
        spike_grad,
        input_channel,
        device,
        input_height,
        input_width,
        rough_pixel,
        threshold=1.0,
        reshape_bool=True,
        beta_learn=True,
        threshold_learn=True,
        reset="subtract",
        power=True,
        repeat_input=False,
        time_aware_loss=False,
    ):
        super().__init__()
        rand.give_seed()
        self.beta_learn = beta_learn
        self.threshold_learn = threshold_learn
        self.reshape_bool = reshape_bool
        self.input_height = input_height
        self.input_width = input_width
        self.rough_pixel = rough_pixel
        self.power = power
        self.input_channel = input_channel
        self.device = device
        self.repeat_input = repeat_input
        self.time_aware_loss = time_aware_loss
        c0 = input_channel
        c1 = 64
        c2 = 128
        c3 = 256
        c4 = 512
        n_class = 1
        encode_kernel = 5
        ratio_dropout = 0.2
        decode_kernel = 5
        n_neuron = 4096
        self.down1 = nn.Sequential(
            nn.Conv2d(c0, c1, encode_kernel, padding=encode_kernel // 2),
            snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=beta_learn,
                learn_threshold=threshold_learn,
                reset_mechanism=reset,
            ),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(ratio_dropout),
        ).to(device)
        self.down2 = nn.Sequential(
            nn.Conv2d(c1, c2, encode_kernel, padding=encode_kernel // 2),
            snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=beta_learn,
                learn_threshold=threshold_learn,
                reset_mechanism=reset,
            ),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(ratio_dropout),
        ).to(device)
        self.down3 = nn.Sequential(
            nn.Conv2d(c2, c3, encode_kernel, padding=encode_kernel // 2),
            snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=beta_learn,
                learn_threshold=threshold_learn,
                reset_mechanism=reset,
            ),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(ratio_dropout),
        ).to(device)
        self.down4 = nn.Sequential(
            nn.Conv2d(c3, c4, encode_kernel, padding=encode_kernel // 2),
            snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=beta_learn,
                learn_threshold=threshold_learn,
                reset_mechanism=reset,
            ),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(ratio_dropout),
        ).to(device)
        self.down5 = nn.Sequential(
            nn.AdaptiveMaxPool2d((rough_pixel, rough_pixel)),
            nn.Conv2d(c4, n_class, 1, padding=0),
            snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=beta_learn,
                learn_threshold=threshold_learn,
                output=True,
                reset_mechanism="none",
            ),
        ).to(device)
        self.network_lst = [self.down1, self.down2, self.down3, self.down4, self.down5]
