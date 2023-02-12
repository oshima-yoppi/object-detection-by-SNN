import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as SF
from snntorch import surrogate
import os
import torch
from . import network


# 設定ファイル。ここでいろんな変数を定義

# blender関連の定数
FOCAL = 0.050# 焦点距離 
IMG_HEIGHT, IMG_WIDTH = 260, 346 # カメラの大きさ[pix]
SENSOR_HEIGHT, SENSOR_WIDTH = 0.026, 0.0346 # イメージセンサの大きさ [m]
CAM_X, CAM_Y, CAM_Z = 64, 64, 164 # カメラの初期位置[m,m,m]


# イベントかめらの極性を分けるかどうか
BOOL_DISTINGUISH_EVENT = True
INPUT_CHANNEL = 2  if BOOL_DISTINGUISH_EVENT else 1
# network関連の定数
INPUT_HEIGHT, INPUT_WIDTH = 130, 173
# INPUT_HEIGHT, INPUT_WIDTH = 65, 86
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BETA = 0.95
BATCH_SIZE = 12
BATCH_SIZE_TEST = 1
PARM_LEARN = False
SPIKE_GRAD = surrogate.atan()
CORRECT_RATE = 0.5
NET = network.FullyConv3(beta=BETA, spike_grad=SPIKE_GRAD, device=DEVICE, input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, parm_learn=PARM_LEARN, input_channel=INPUT_CHANNEL)



# path
VIDEO_PATH = 'blender/video'
DEM_NP_PATH = 'blender/dem' 
ACCUMULATE_EVENT_MILITIME = 100 #[ms] # 何msイベントをためるか
ACCUMULATE_EVENT_MICROTIME= ACCUMULATE_EVENT_MILITIME*1000 #[us]
DATASET_PATH = 'dataset' # datasetのパス
# DATASET_ACCEVENT_PATH = os.path.join(DATASET_PATH, str(ACCUMULATE_EVENT_MICROTIME)) # dataset/〇〇  ←何秒ためるかを表す
EVENT_TH = 0.5# イベントカメラの閾値
RAW_EVENT_PATH = f'data/th-{str(EVENT_TH)}' # v2eから出力されたイベント生データ
PROCESSED_EVENT_DATASET_PATH = f'dataset/{ACCUMULATE_EVENT_MICROTIME}_({INPUT_HEIGHT},{INPUT_WIDTH})_th-{EVENT_TH}'
MODEL_PATH = 'models/model1.pth'





