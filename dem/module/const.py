import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as SF
from snntorch import surrogate

import torch
from . import network


# 設定ファイル。ここでいろんな変数を定義

# blender
FOCAL = 0.050# 焦点距離 
IMG_HEIGHT, IMG_WIDTH = 260, 346 # カメラの大きさ[pix]
SENSOR_HEIGHT, SENSOR_WIDTH = 0.026, 0.0346 # イメージセンサの大きさ [m]
CAM_X, CAM_Y, CAM_Z = 64, 64, 164 # カメラの初期位置[m,m,m]

# path
# video_folder = 


# network
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BETA = 0.95
PARM_LEARN = True
SPIKE_GRAD = surrogate.atan()
NET = network.FullyConv3(beta=BETA, spike_grad=SPIKE_GRAD, device=DEVICE, parm_learn=PARM_LEARN)
