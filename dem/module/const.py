import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as SF
from snntorch import surrogate
import os
import torch
from . import network
from .const_blender import *

# 設定ファイル。ここでいろんな変数を定義



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
LR = 1e-3
CORRECT_RATE = 0.5
NET = network.FullyConv3(beta=BETA, spike_grad=SPIKE_GRAD, device=DEVICE, input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, parm_learn=PARM_LEARN, input_channel=INPUT_CHANNEL)



LABEL_PATH = 'label_only'



ACCUMULATE_EVENT_MILITIME = 100 #[ms] # 何msイベントをためるか
ACCUMULATE_EVENT_MICROTIME= ACCUMULATE_EVENT_MILITIME*1000 #[us]
DATASET_PATH = 'dataset' # datasetのパス
# DATASET_ACCEVENT_PATH = os.path.join(DATASET_PATH, str(ACCUMULATE_EVENT_MICROTIME)) # dataset/〇〇  ←何秒ためるかを表す
EVENT_TH = 0.1# イベントカメラの閾値
RAW_EVENT_PATH = f'raw-data/th-{str(EVENT_TH)}' # v2eから出力されたイベント生データ
PROCESSED_EVENT_DATASET_PATH = f'dataset/{ACCUMULATE_EVENT_MICROTIME}_({INPUT_HEIGHT},{INPUT_WIDTH})_th-{EVENT_TH}'

NETWORK_CLASS_NAME = NET.__class__.__name__
MODEL_NAME = f'{NETWORK_CLASS_NAME}_{ACCUMULATE_EVENT_MICROTIME}_({INPUT_HEIGHT},{INPUT_WIDTH})_th-{EVENT_TH}'
MODEL_PATH = f'models/{MODEL_NAME}.pth'

RESULT_PATH = f'result_img/{MODEL_NAME}'



# print(NETWORK_CLASS_NAME)

