# -*- coding: utf-8 -*-

import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchvision
import numpy as np
from torch import cuda

import numpy 
from . import step_func
#import step_func

class Conv_GPU(nn.Module):
    """
    Args:
        n_in (int): The number of input.
        n_out (int): The number of output.
        l_tau (floot): Degree of leak (From 0 to 1).
        soft (bool): Change output activation to sigmoid func (True)
                     or Step func. (False)
        rec (bool): Adding recurrent connection or not.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, l_tau=0.8, soft=False, rec=False, forget=False, dual=False,nobias=False, initial_bias=-0.5, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.l_tau = l_tau
        self.rec = rec
        self.forget = forget
        self.dual = dual
        self.soft = soft
        self.s = None
        self.y = None
        self.initial_bias = initial_bias
        print("==== self.rec ====",rec)
        print("=== GPU ===",self.gpu)
        print("==== self.forget ====",self.forget)
        print(" ==== dual Gate ====",self.dual)
        

        self.Wx = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) #入力チャネル数, 出力チャネル数, フィルタサイズ
        # torch.nn.init.xavier_uniform_(self.Wx.weight)
        #print("self.rec in Conv_SNU",self.rec)
        if rec:
            print("recだよー")
            self.Wy = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wy.weight)
            self.Wi = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wi.weight)
            self.Ri = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Ri.weight)
        if forget:
            # 膜電位忘却ゲート
            #print("forgetだよー")
            self.Wf = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wf.weight,0.1)
            self.Rf = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Rf.weight,0.1)
        if dual:
            # スパイク再突入　＋　膜電位忘却ゲート
            self.Wy = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wy.weight)
            self.Wi = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wi.weight)
            self.Ri = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Ri.weight)
            self.Wf = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wf.weight,0.1)
            self.Rf = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Rf.weight,0.1)

        if nobias:
            self.b = None
        else:
            self.b = nn.Parameter(torch.Tensor([initial_bias]))

    def reset_state(self, s=None, y=None):
        self.s = s
        self.y = y

    def initialize_state(self, shape): #shape (バッチ,tチャネル,oh,ow)
        dtype = torch.float
        self.oh = int(((shape[2] + 2*self.padding - self.kernel_size)/self.stride) + 1) # OH=H+2*P-FH/s +1
        self.ow = int(((shape[3] + 2*self.padding - self.kernel_size)/self.stride) + 1)
        ###########\dem_conv_classification.py
        self.s = torch.zeros((shape[0], self.out_channels, self.oh, self.ow),dtype=dtype)
        self.y = torch.zeros((shape[0], self.out_channels, self.oh, self.ow),dtype=dtype)
        ############dem_autoencoder_segmentation.py
        #self.s = torch.zeros((shape[0], self.out_channels, shape[2], shape[3]),device=device,dtype=dtype)
        #self.y = torch.zeros((shape[0], self.out_channels, shape[2], shape[3]),device=device,dtype=dtype)
        #self.Wrs = nn.Parameter(torch.empty((shape[0], self.out_channels, self.oh, self.ow),device=device,dtype=dtype))
        #self.br = nn.Parameter(torch.empty((shape[0], self.out_channels, self.oh, self.ow),device=device,dtype=dtype))
    
    def forward(self,x):
        if self.s is None:
            self.initialize_state(x.shape)

        if type(self.s) == numpy.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()
    
        #print('=self.Wy(self.y)',self.Wy(self.y).shape)
        #print('=self.Wx(x)',self.Wx(x).shape)
        if self.rec:
            # print("rec yessss")
            #f = torch.sigmoid(self.Wf(x) + self.Rf(self.y))
            # spike 再入力ゲート
            i = torch.sigmoid(self.Wi(x) + self.Ri(self.y))
            s = F.elu(abs(self.Wx(x)) + i*self.Wy(self.y) + self.l_tau * self.s * (1-self.y))
        if self.forget:
            #print("forget yesssss")
            # 膜電位忘却ゲート
            f = torch.sigmoid(self.Wf(x) + self.Rf(self.y))
            s = F.elu(abs(self.Wx(x)) + (self.l_tau-f) * self.s * (1-self.y))
        if self.dual:
            #print("dual Gate yesssss")
            i = torch.sigmoid(self.Wi(x) + self.Ri(self.y))
            f = torch.sigmoid(self.Wf(x) + self.Rf(self.y))
            s = F.elu(abs(self.Wx(x)) + i*self.Wy(self.y) + (self.l_tau-f) * self.s * (1-self.y))
            #print('i',i.shape)
            #print('f',f.shape)
            #print('s',s.shape)
        else:
            #print("rec Noooooo")
            # s = F.elu(abs(self.Wx(x)) + self.l_tau * self.s * (1-self.y))
            s = F.elu(self.Wx(x) + self.l_tau * self.s * (1-self.y))
            # s = self.Wx(x) + self.l_tau * self.s * (1-self.y)
        #s = F.elu(abs(self.Wx(x)) + r * self.s * (1-self.y))

        if self.soft:

            axis = 1
            bias_ = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)]
            #print("bias_:",bias_)
            y = torch.sigmoid(bias_)
        else:
            axis = 0
            bias = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)] #error!! two types
            bias = s + self.b
            y = step_func.spike_fn(bias)
        
        self.s = s
        self.y = y

        return y

class SNU(nn.Module):
    def __init__(self, in_channels, out_channels, l_tau=0.8, soft=False, rec=False, nobias=False, initial_bias=-0.5, gpu=True, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels= out_channels
        self.l_tau = l_tau
        self.rec = rec
        self.soft = soft
        self.gpu = gpu
        self.bias = bias
        self.s = None
        self.y = None
        self.initial_bias = initial_bias

        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
        
        
        self.Wx = nn.Linear(in_channels, out_channels, bias=False)
        if self.rec:
            print("recだよ")
            self.Wrec = nn.Linear(out_channels, out_channels, bias=self.bias)
    

        if not self.bias:
            self.b = None
        else:

            device = torch.device(device)
            
            self.b = nn.Parameter(torch.Tensor([initial_bias]))
            # print("self.b",self.b)
            # print('afk;lasdl;kaskl;l;kj')
                            
    def reset_state(self, s=None, y=None):
        self.s = s
        self.y = y

    def initialize_state(self, shape):
        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
            
        self.s = torch.zeros((shape[0], self.out_channels),device=device,dtype=dtype)
        self.y = torch.zeros((shape[0], self.out_channels),device=device,dtype=dtype)
              
    
    def forward(self,x):
        if self.s is None:
            #print("self.s is none")
            self.initialize_state(x.shape)


        if type(self.s) == numpy.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()
    
        if self.rec:
            s = F.elu(self.Wx(x) + self.l_tau * self.s * (1-self.y) + self.Wrec(self.y))
            
        else:
            s = F.elu(self.Wx(x) + self.l_tau * self.s * (1-self.y))
        
        if self.soft:

            axis = 1
            bias_ = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)]
            #print("bias_:",bias_)
            y = torch.sigmoid(bias_)###元々シグモイド関数。数値回帰のためにeluに変更
            # y = F.elu(bias_)

        else:
            axis = 0
            bias = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)] #error!! two types
            bias = s + self.b
            y = step_func.spike_fn(bias)
        # print(self.s)
        self.s = s
        self.y = y

        return y

class SNU_None(nn.Module):
    def __init__(self, in_channels, out_channels, l_tau=0.8, soft=False, rec=False, nobias=False, initial_bias=-0.5, gpu=True, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels= out_channels
        self.l_tau = l_tau
        self.rec = rec
        self.soft = soft
        self.gpu = gpu
        self.s = None
        self.y = None
        self.initial_bias = initial_bias

        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
        
        #self.w1 = torch.empty((n_in, n_out),  device=device, dtype=dtype, requires_grad=True)
        #torch.nn.init.normal_(self.w1, mean=0.0)
        
        #self.Wx = torch.einsum("abc,cd->abd", (x_data, w1))
        #self.Wx = nn.Linear(4374, out_channels, bias=False).to(device)
        self.Wx = nn.Linear(in_channels, out_channels, bias=False).to(device)
        # print(self.Wx)
        # nn.init.uniform_(self.Wx.weight, -0.1, 0.1) #3.0
        # torch.nn.init.xavier_uniform_(self.Wx.weight)
        # print('77777777777777777')
        # print(self.Wx)
    

        if nobias:
            self.b = None
        else:

            #print("initial_bias",initial_bias)
            device = torch.device(device)
            
            self.b = nn.Parameter(torch.Tensor([initial_bias]).to(device))
            #print("self.b",self.b)
                            
    def reset_state(self, s=None, y=None):
        self.s = s
        self.y = y

    def initialize_state(self, shape):
        if self.gpu:
            #xp = cuda.cupy
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
            
        self.s = torch.zeros((shape[0], self.out_channels),device=device,dtype=dtype)
        self.y = torch.zeros((shape[0], self.out_channels),device=device,dtype=dtype)
              
    
    def forward(self,x):
        if self.s is None:
            #print("self.s is none")
            self.initialize_state(x.shape)


        if type(self.s) == numpy.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()
    
        #print("x in snu.shape",x.shape) #x in snu.shape torch.Size([256, 784])        
        #print("self.Wx(x).shape",self.Wx(x).shape)
        #print("self.s.shape : ",self.s.shape)
        # s = F.elu(abs(self.Wx(x)) + self.l_tau * self.s * (1-self.y))
        # s = F.elu(self.Wx(x) + self.l_tau * self.s * (1-self.y))
        # print(f'wx.shape:{self.Wx.shape}')
        ####出量の回帰をなくしてみた。畳み込み積分の再現
        s = self.Wx(x) + self.l_tau * self.s 
        # print("s : ",s)

        # if self.soft:

        #     axis = 1
        #     bias_ = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)]
        #     #print("bias_:",bias_)
        #     y = torch.sigmoid(bias_)###元々シグモイド関数。数値回帰のためにeluに変更
        #     # y = F.elu(bias_)

        # else:
        #     axis = 0
        #     #print("s.shape:", s.shape)
        #     #print("self.b.shape:", self.b.shape)
        #     #print("self.initial_bias.shape:",self.initial_bias.shape)
        #     # print("self.b.shape !!!!!!!!!!!!!!!! ", self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)].shape)
        #     bias = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)] #error!! two types
        #     # print("bias:",bias)
        #     #print("s in snu:",s)
        #     bias = s + self.b
        #     # print(bias)

        #     y = step_func.spike_fn(bias-0.3)
        # print(self.s)
        y = s
        self.s = s
        self.y = y

        return y

class Conv_SNU(nn.Module):
    """
    Args:
        n_in (int): The number of input.
        n_out (int): The number of output.
        l_tau (floot): Degree of leak (From 0 to 1).
        soft (bool): Change output activation to sigmoid func (True)
                     or Step func. (False)
        rec (bool): Adding recurrent connection or not.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, l_tau=0.8, soft=False, rec=False, forget=False, dual=False,nobias=False, initial_bias=-0.5, gpu=True, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.l_tau = l_tau
        self.rec = rec
        self.forget = forget
        self.dual = dual
        self.soft = soft
        self.gpu = gpu
        self.s = None
        self.y = None
        self.initial_bias = initial_bias
        print("==== self.rec ====",rec)
        print("=== GPU ===",self.gpu)
        print("==== self.forget ====",self.forget)
        print(" ==== dual Gate ====",self.dual)
        if self.gpu:
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")

        self.Wx = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
        # torch.nn.init.xavier_uniform_(self.Wx.weight)
        #print("self.rec in Conv_SNU",self.rec)
        if rec:
            print("recだよー")
            self.Wy = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wy.weight)
            self.Wi = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wi.weight)
            self.Ri = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Ri.weight)
        if forget:
            # 膜電位忘却ゲート
            #print("forgetだよー")
            self.Wf = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wf.weight,0.1)
            self.Rf = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Rf.weight,0.1)
        if dual:
            # スパイク再突入　＋　膜電位忘却ゲート
            self.Wy = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wy.weight)
            self.Wi = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wi.weight)
            self.Ri = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Ri.weight)
            self.Wf = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Wf.weight,0.1)
            self.Rf = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False).to(device) #入力チャネル数, 出力チャネル数, フィルタサイズ
            torch.nn.init.xavier_uniform_(self.Rf.weight,0.1)

        if nobias:
            self.b = None
        else:
            device = torch.device(device)
            self.b = nn.Parameter(torch.Tensor([initial_bias]).to(device))

    def reset_state(self, s=None, y=None):
        self.s = s
        self.y = y

    def initialize_state(self, shape): #shape (バッチ,tチャネル,oh,ow)
        if self.gpu:
            dtype = torch.float
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            dtype = torch.float
            device=torch.device("cpu")
        self.oh = int(((shape[2] + 2*self.padding - self.kernel_size)/self.stride) + 1) # OH=H+2*P-FH/s +1
        self.ow = int(((shape[3] + 2*self.padding - self.kernel_size)/self.stride) + 1)
        ###########\dem_conv_classification.py
        self.s = torch.zeros((shape[0], self.out_channels, self.oh, self.ow),device=device,dtype=dtype)
        self.y = torch.zeros((shape[0], self.out_channels, self.oh, self.ow),device=device,dtype=dtype)
        ############dem_autoencoder_segmentation.py
        #self.s = torch.zeros((shape[0], self.out_channels, shape[2], shape[3]),device=device,dtype=dtype)
        #self.y = torch.zeros((shape[0], self.out_channels, shape[2], shape[3]),device=device,dtype=dtype)
        #self.Wrs = nn.Parameter(torch.empty((shape[0], self.out_channels, self.oh, self.ow),device=device,dtype=dtype))
        #self.br = nn.Parameter(torch.empty((shape[0], self.out_channels, self.oh, self.ow),device=device,dtype=dtype))
    
    def forward(self,x):
        if self.s is None:
            self.initialize_state(x.shape)

        if type(self.s) == numpy.ndarray:
            self.s = torch.from_numpy(self.s.astype(np.float32)).clone()
    
        #print('=self.Wy(self.y)',self.Wy(self.y).shape)
        #print('=self.Wx(x)',self.Wx(x).shape)
        if self.rec:
            # print("rec yessss")
            #f = torch.sigmoid(self.Wf(x) + self.Rf(self.y))
            # spike 再入力ゲート
            i = torch.sigmoid(self.Wi(x) + self.Ri(self.y))
            s = F.elu(abs(self.Wx(x)) + i*self.Wy(self.y) + self.l_tau * self.s * (1-self.y))
        if self.forget:
            #print("forget yesssss")
            # 膜電位忘却ゲート
            f = torch.sigmoid(self.Wf(x) + self.Rf(self.y))
            s = F.elu(abs(self.Wx(x)) + (self.l_tau-f) * self.s * (1-self.y))
        if self.dual:
            #print("dual Gate yesssss")
            i = torch.sigmoid(self.Wi(x) + self.Ri(self.y))
            f = torch.sigmoid(self.Wf(x) + self.Rf(self.y))
            s = F.elu(abs(self.Wx(x)) + i*self.Wy(self.y) + (self.l_tau-f) * self.s * (1-self.y))
            #print('i',i.shape)
            #print('f',f.shape)
            #print('s',s.shape)
        else:
            #print("rec Noooooo")
            # s = F.elu(abs(self.Wx(x)) + self.l_tau * self.s * (1-self.y))
            s = F.elu(self.Wx(x) + self.l_tau * self.s * (1-self.y))
            # s = self.Wx(x) + self.l_tau * self.s * (1-self.y)
        #s = F.elu(abs(self.Wx(x)) + r * self.s * (1-self.y))

        if self.soft:

            axis = 1
            bias_ = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)]
            #print("bias_:",bias_)
            y = torch.sigmoid(bias_)
        else:
            axis = 0
            bias = s + self.b[(...,) + (None,) * (s.ndim - self.b.ndim - axis)] #error!! two types
            bias = s + self.b
            y = step_func.spike_fn(bias)
        
        self.s = s
        self.y = y

        return y


