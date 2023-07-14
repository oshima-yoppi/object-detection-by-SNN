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


class BaseFunction(nn.Module):
    def __call__(self, data, time):
        # soft = nn.Softmax2d()
        spk_rec = []
        utils.reset(self.network)  # resets hidden states for all LIF neurons in net
        # print(data.shape)

        for step in range(time):  # data.size(0) = number of time steps
            spk_out, mem_out = self.network(data[step])
            # print(spk_out.shape)
            if self.reshape_bool:
                batch = len(spk_out)
                spk_out = spk_out.reshape(batch, 2, self.input_height, self.input_width)

            spk_rec.append(spk_out)

        spk_rec = torch.stack(spk_rec)
        # print(spk_rec.shape)
        spk_cnt = compute_loss.spike_count(
            spk_rec, channel=True
        )  # batch channel(n_class) pixel pixel
        spk_cnt_ = (
            spk_cnt[0, 0, :, :]
            .reshape(self.input_height, self.input_width)
            .to("cpu")
            .detach()
            .numpy()
            .copy()
        )

        # print(np.sum(spk_cnt_.reshape(-1)))

        pred_pro = F.softmax(spk_cnt, dim=1)
        pred_pro_ = (
            pred_pro[0, 0, :, :]
            .reshape(self.input_height, self.input_width)
            .to("cpu")
            .detach()
            .numpy()
            .copy()
        )
        pred_pro__ = (
            pred_pro[0, 1, :, :]
            .reshape(self.input_height, self.input_width)
            .to("cpu")
            .detach()
            .numpy()
            .copy()
        )

        return pred_pro


class FullyConv2_new(nn.Module):
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
        encode_kernel = 5
        decode_kernel = 5
        n_neuron = 4096

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
        self.middle = nn.Sequential(
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
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c3, c2, decode_kernel, padding=encode_kernel // 2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c2, c0, decode_kernel, padding=decode_kernel // 2),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)
        self.output = nn.Sequential(
            nn.Conv2d(c0, n_class, 1,),
            nn.AdaptiveMaxPool2d((self.input_height, self.input_width)),
            snn.Leaky(
                beta=beta,
                spike_grad=spike_grad,
                init_hidden=True,
                output=True,
                learn_beta=parm_learn,
                learn_threshold=parm_learn,
                reset_mechanism=reset,
            ),
        ).to(device)

        self.network_lst = [
            self.down1,
            self.down2,
            self.middle,
            self.up1,
            self.up2,
            self.output,
        ]

    def forward(self, data, time):
        self.spike_count = 0
        spk_rec = []
        for net in self.network_lst:
            utils.reset(net)
        for step in range(time):
            d1 = self.down1(data[step])
            d2 = self.down2(d1)
            m1 = self.middle(d2)
            u1 = self.up1(m1)
            u2 = self.up2(u1)
            output, _ = self.output(u2)
            spk_rec.append(output)
            if self.power:
                self.spike_count += torch.sum(d1)
                self.spike_count += torch.sum(d2)
                self.spike_count += torch.sum(m1)
                self.spike_count += torch.sum(u1)
                self.spike_count += torch.sum(u1)
                self.spike_count += torch.sum(output)
        # print(self.spike_count)
        spk_rec = torch.stack(spk_rec)
        # print(spk_rec.shape)
        spk_cnt = compute_loss.spike_count(
            spk_rec, channel=True
        )  # batch channel(n_class) pixel pixel
        spk_cnt_ = (
            spk_cnt[0, 0, :, :]
            .reshape(self.input_height, self.input_width)
            .to("cpu")
            .detach()
            .numpy()
            .copy()
        )

        # print(np.sum(spk_cnt_.reshape(-1)))

        pred_pro = F.softmax(spk_cnt, dim=1)
        # pred_pro_ = pred_pro[0,0,:,:].reshape(self.input_height, self.input_width).to('cpu').detach().numpy().copy()
        # pred_pro__ = pred_pro[0,1,:,:].reshape(self.input_height, self.input_width).to('cpu').detach().numpy().copy()

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
            nn.Conv2d(c0, n_class, 1,),
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
            nn.Conv2d(c0, n_class, 1,),
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
            nn.Conv2d(c1, n_class, 1,),
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
