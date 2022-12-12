# -*- coding: utf-8 -*-
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from . import snu_layer

import numpy as np
# from torchsummary import summary
import matplotlib.pyplot as plt


class NetGpu(torch.nn.Module):
    def __init__(self, num_time=10, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, power=False, gpu=True,
                 batch_size=8, reg_n = 65536, pixel=64 ):
        super().__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.pixel = pixel
        self.dual = dual
        self.power = power
        n1 = 1024
        n2 = 512
        n3 = 3
        # Encoder layers
        self.l1 = snu_layer.Conv_SNU(in_channels=1, out_channels=4, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual)
        self.l2 = snu_layer.Conv_SNU(in_channels=4, out_channels=16, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual)
        
        # self.out1 = snu_layer.SNU_None(n1, n3, l_tau=l_tau, soft=soft, gpu=gpu)
        # self.l4 = snu_layer.SNU_None(n2, n3, l_tau=l_tau, soft=soft, gpu=gpu)

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        # self.out1.reset_state()

   
        
    def forward(self, x, y):
        """
        x: inputs
        """
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        record = []
        #print('out shape',out.shape)
        self._reset_state()
    

        for t in range(self.num_time):
            x_t = x[:, t, :, :]  #batch, time, x, y
            x_t = x_t.reshape((len(x_t), 1, self.pixel, self.pixel))#torch.Size([6, 2, 128, 128])
            x_ = self.l1(x_t) 
            x_ = F.max_pool2d(x_, 2) 
            x_ = self.l2(x_) 
            x_ = F.max_pool2d(x_, 2) 
            x_ = x_.view(len(x_), -1)
            x_ = self.out1(x_)
            # print(x_.shape) #torch.Size([8, 3])
            # x_ *= 50
            x_ = x_.unsqueeze(dim=2)
            record.append(x_)
            
        out_rec = torch.cat(record, dim = 2)
        # print('----------------------')
        # print(out_rec.shape)#torch.Size([Batch, 3, 20]) (batchsize, xyz, time)
        # print(out_rec)
        
        return out_rec

# 改(05/08~)よｐっぴver　
class VectorRegression(torch.nn.Module):
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, power=False, gpu=True,
                 batch_size=128, reg_n = 65536 ):
        super().__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual
        self.power = power
        n1 = 1024
        n2 = 512
        n3 = 3
        # Encoder layers
        self.l1 = snu_layer.Conv_SNU(in_channels=2, out_channels=4, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l2 = snu_layer.Conv_SNU(in_channels=4, out_channels=16, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        # self.l3 = snu_layer.Conv_SNU(in_channels=16, out_channels=32, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        # self.l4 = snu_layer.Conv_SNU(in_channels=32, out_channels=64, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        # self.l3 = snu_layer.SNU(n1, n2, l_tau=l_tau, soft=soft, gpu=gpu)
        # self.l4 = snu_layer.SNU(n2, n3, l_tau=l_tau, soft=soft, gpu=gpu)
        # self.l3 = nn.Linear(n1, n2, bias = True)
        # self.l4 = nn.Linear(n2, n3, bias = True)
        self.out1 = snu_layer.SNU_None(n1, n3, l_tau=l_tau, soft=soft, gpu=gpu)
        # self.l4 = snu_layer.SNU_None(n2, n3, l_tau=l_tau, soft=soft, gpu=gpu)

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.out1.reset_state()

   
        
    def forward(self, x):
        """
        x: inputs
        """
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        record = []
        #print('out shape',out.shape)
        self._reset_state()
    

        for t in range(self.num_time):
            x_t = x[:, :, t, :, :]  
            x_t = x_t.reshape((len(x_t), 2, 128, 128))#torch.Size([6, 2, 128, 128])
            x_ = self.l1(x_t) 
            x_ = F.max_pool2d(x_, 2) 
            x_ = self.l2(x_) 
            x_ = F.max_pool2d(x_, 2) 
            x_ = x_.view(len(x_), -1)
            x_ = self.out1(x_)
            # print(x_.shape) #torch.Size([8, 3])
            # x_ *= 50
            x_ = x_.unsqueeze(dim=2)
            record.append(x_)
            
        out_rec = torch.cat(record, dim = 2)
        # print('----------------------')
        # print(out_rec.shape)#torch.Size([Batch, 3, 20]) (batchsize, xyz, time)
        # print(out_rec)
        
        return out_rec


## conv 4層
class RSNU(torch.nn.Module):
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=True, forget=False, dual=False, bias=False, gpu=True,
                 batch_size=128):
        super().__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual
        self.bias = bias
        c1, c2, c3, c4,c5 = 2*128*128, 64*64, 32*32, 256, 64
        o1 = 64
        o2 = 64
        # n1 = 4096
        n2 = 512
        n3 = 1
        # Encoder layers
        self.l1 = snu_layer.SNU(in_channels=c1, out_channels=c2, l_tau=l_tau, soft=soft, rec=self.rec, gpu=gpu, bias=self.bias)
        self.l2 = snu_layer.SNU(in_channels=c2, out_channels=c3, l_tau=l_tau, soft=soft, rec=self.rec, gpu=gpu, bias=self.bias)
        self.l3 = snu_layer.SNU(in_channels=c3, out_channels=c4, l_tau=l_tau, soft=soft, rec=self.rec, gpu=gpu, bias=self.bias)
        self.l4 = snu_layer.SNU(in_channels=c4, out_channels=c5, l_tau=l_tau, soft=soft, rec=self.rec, gpu=gpu, bias=self.bias)
        
        self.out1 = snu_layer.SNU_None(o1, o2, l_tau=l_tau, soft=soft, gpu=gpu, bias=self.bias)
        
    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()
        self.out1.reset_state()

   
        
    def forward(self, x, y):
        """
        x: inputs
        y: labels
        """
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        record = []
        #print('out shape',out.shape)
        self._reset_state()
    

        for t in range(self.num_time):
            x_t = x[:, :, t, :, :]  
            x_t = x_t.reshape((len(x_t), 2, 128, 128))#torch.Size([6, 2, 128, 128])
            # print(f'x_t.shape{x_t.shape}')
            # print(f'len(x_t):{len(x_t)}')
            x_t = x_t.reshape(len(x_t), -1)
            x_ = self.l1(x_t) 
            x_ = self.l2(x_) 
            x_ = self.l3(x_) 
            x_ = self.l4(x_) 
            # print(f'x_.shape:{x_.shape},{len(x_)}')#x_.shape:torch.Size([batch, 128, 8, 8]),batch
            # x_ = x_.view(self.batch_size, -1)
            x_ = x_.view(len(x_), -1)
            # print(f'x_.shape:{x_.shape}')#trch.Size([BatchSize, 16*64*64?])
            x_ = self.out1(x_)
            x_ = torch.mean(x_, dim=1)
            x_ = x_.view(len(x_), -1)
            # print(x_)
            record.append(x_)
            
        out_rec = torch.cat(record, dim = 1)
        # print(out_rec.shape)#torch.Size([6, 100]) (batchsize, time)型のω
        # print(out_rec)
        
        return out_rec

## conv 4層
class Conv4Regression(torch.nn.Module):
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, bias=False, gpu=True,
                 batch_size=128 ):
        super().__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual
        self.bias = bias
        c1, c2, c3, c4,c5 = 2, 16, 32, 64, 128
        n1 = 8192
        # n1 = 4096
        n2 = 512
        n3 = 1
        # Encoder layers
        self.l1 = snu_layer.Conv_SNU(in_channels=c1, out_channels=c2, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu, bias=self.bias)
        self.l2 = snu_layer.Conv_SNU(in_channels=c2, out_channels=c3, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu, bias=self.bias)
        self.l3 = snu_layer.Conv_SNU(in_channels=c3, out_channels=c4, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu, bias=self.bias)
        self.l4 = snu_layer.Conv_SNU(in_channels=c4, out_channels=c5, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu, bias=self.bias)
        # self.l3 = snu_layer.Conv_SNU(in_channels=16, out_channels=32, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        # self.l4 = snu_layer.Conv_SNU(in_channels=32, out_channels=64, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        # self.l3 = snu_layer.SNU(n1, n2, l_tau=l_tau, soft=soft, gpu=gpu)
        # self.l4 = snu_layer.SNU(n2, n3, l_tau=l_tau, soft=soft, gpu=gpu)
        # self.l3 = nn.Linear(n1, n2, bias = True)
        # self.l4 = nn.Linear(n2, n3, bias = True)
        self.out1 = snu_layer.SNU_None(n1, n1, l_tau=l_tau, soft=soft, gpu=gpu, bias=self.bias)
        # self.l4 = snu_layer.SNU_None(n2, n3, l_tau=l_tau, soft=soft, gpu=gpu)

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.l4.reset_state()
        self.out1.reset_state()
        # self.l4.reset_state()

   
        
    def forward(self, x, y):
        """
        x: inputs
        y: labels
        """
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        record = []
        #print('out shape',out.shape)
        self._reset_state()
    

        for t in range(self.num_time):
            x_t = x[:, :, t, :, :]  
            x_t = x_t.reshape((len(x_t), 2, 128, 128))#torch.Size([6, 2, 128, 128])
            x_ = self.l1(x_t) 
            x_ = self.l2(x_) 
            x_ = self.l3(x_) 
            x_ = self.l4(x_) 
            # print(f'x_.shape:{x_.shape},{len(x_)}')#x_.shape:torch.Size([batch, 128, 8, 8]),batch
            # x_ = x_.view(self.batch_size, -1)
            x_ = x_.view(len(x_), -1)
            # print(f'x_.shape:{x_.shape}')#trch.Size([BatchSize, 16*64*64?])
            x_ = self.out1(x_)
            x_ = torch.mean(x_, dim=1)
            x_ = x_.view(len(x_), -1)
            # print(x_)
            record.append(x_)
            
        out_rec = torch.cat(record, dim = 1)
        # print(out_rec.shape)#torch.Size([6, 100]) (batchsize, time)型のω
        # print(out_rec)
        
        return out_rec


# 改(05/08~)よｐっぴver　
class SNU_Regression(torch.nn.Module):
    def __init__(self, num_time=20, l_tau=0.8, soft=False, rec=False, forget=False, dual=False, power=False, gpu=True,
                 batch_size=128, reg_n = 65536 ):
        super().__init__()

        
        self.num_time = num_time
        self.batch_size = batch_size
        self.rec = rec
        self.forget = forget
        self.dual = dual
        self.power = power
        n1 = 1024
        n2 = 512
        n3 = 1
        # Encoder layers
        self.l1 = snu_layer.Conv_SNU(in_channels=2, out_channels=4, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        self.l2 = snu_layer.Conv_SNU(in_channels=4, out_channels=16, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        # self.l3 = snu_layer.Conv_SNU(in_channels=16, out_channels=32, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        # self.l4 = snu_layer.Conv_SNU(in_channels=32, out_channels=64, kernel_size=3, padding=1, l_tau=l_tau, soft=soft, rec=self.rec, forget=self.forget, dual=self.dual, gpu=gpu)
        # self.l3 = snu_layer.SNU(n1, n2, l_tau=l_tau, soft=soft, gpu=gpu)
        # self.l4 = snu_layer.SNU(n2, n3, l_tau=l_tau, soft=soft, gpu=gpu)
        # self.l3 = nn.Linear(n1, n2, bias = True)
        # self.l4 = nn.Linear(n2, n3, bias = True)
        self.out1 = snu_layer.SNU_None(n1, n3, l_tau=l_tau, soft=soft, gpu=gpu)
        # self.l4 = snu_layer.SNU_None(n2, n3, l_tau=l_tau, soft=soft, gpu=gpu)

    def _reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()
        # self.l3.reset_state()
        # self.l4.reset_state()
        self.out1.reset_state()
        # self.l4.reset_state()

   
        
    def forward(self, x, y):
        """
        x: inputs
        y: labels
        """
        loss = None
        correct = 0
        sum_out = None
        dtype = torch.float
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        record = []
        #print('out shape',out.shape)
        self._reset_state()
    

        for t in range(self.num_time):
            x_t = x[:, :, t, :, :]  
            x_t = x_t.reshape((len(x_t), 2, 128, 128))#torch.Size([6, 2, 128, 128])
            x_ = self.l1(x_t) 
            x_ = F.max_pool2d(x_, 2) 
            x_ = self.l2(x_) 
            x_ = F.max_pool2d(x_, 2) 
            # x_ = self.l3(x_)
            # x_ = self.l4(x_)
            x_ = x_.view(len(x_), -1)
            # print(f'x_.shape:{x_.shape}')#trch.Size([BatchSize, 16*64*64?])
            x_ = self.out1(x_)
            # x_ = self.l4(x_)
            # x_ *= 50
            record.append(x_)
            
        out_rec = torch.cat(record, dim = 1)
        # print(out_rec.shape)#torch.Size([6, 100]) (batchsize, time)型のω
        # print(out_rec)
        
        return out_rec


