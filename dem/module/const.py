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
FINISH_STEP = 8 # 8
soft_reset =True
if soft_reset:
    RESET = 'subtract'
else:
    RESET = 'zero'
PARM_LEARN = False
TIME_CHANGE = False
SPIKE_GRAD = surrogate.atan()
LR = 1e-3
CORRECT_RATE = 0.5
LOSS_RATE = 1e-7
NET = network.FullyConv2_new(beta=BETA, spike_grad=SPIKE_GRAD, device=DEVICE, input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, parm_learn=PARM_LEARN, input_channel=INPUT_CHANNEL, power=True, reset=RESET)



LABEL_PATH = 'label_only'
LABEL_BOULDER_PATH = "label_only_boulder"



# ACCUMULATE_EVENT_MILITIME = 100 #[ms] # 何msイベントをためるか
ACCUMULATE_EVENT_MILITIME = 25 #[ms] # 何msイベントをためるか
ACCUMULATE_EVENT_MICROTIME= ACCUMULATE_EVENT_MILITIME*1000 #[us]
DATASET_PATH = 'dataset' # datasetのパス
# DATASET_ACCEVENT_PATH = os.path.join(DATASET_PATH, str(ACCUMULATE_EVENT_MICROTIME)) # dataset/〇〇  ←何秒ためるかを表す
EVENT_TH = 0.15# イベントカメラの閾値
RAW_EVENT_PATH = f'raw-data/th-{str(EVENT_TH)}' # v2eから出力されたイベント生データ
RAW_EVENT_ONLY_BOULDER_PATH = f'raw-data_only_boulder/th-{str(EVENT_TH)}' 
PROCESSED_EVENT_DATASET_PATH = f'dataset/{ACCUMULATE_EVENT_MICROTIME}_({INPUT_HEIGHT},{INPUT_WIDTH})_th-{EVENT_TH}_FinTime-{FINISH_STEP}'
PROCESSED_EVENT_DATASET_ONLY_BOULDER_PATH = f'dataset_boulder/{ACCUMULATE_EVENT_MICROTIME}_({INPUT_HEIGHT},{INPUT_WIDTH})_th-{EVENT_TH}'
ANN_DATASET_PATH = "dataset_ann"


NETWORK_CLASS_NAME = NET.__class__.__name__
MODEL_NAME = f'{NETWORK_CLASS_NAME}_{ACCUMULATE_EVENT_MICROTIME}_({INPUT_HEIGHT},{INPUT_WIDTH})_th-{EVENT_TH}_para-{PARM_LEARN}_TimeChange-{TIME_CHANGE}_FinishTime-{FINISH_STEP}_Reset-{RESET}'
MODEL_PATH = f'models/{MODEL_NAME}.pth'

RESULT_PATH = f'result_img/{MODEL_NAME}'


RESULT_ONLY_BOULDER_PATH = f'result_img_boulder/{MODEL_NAME}'



# print(NETWORK_CLASS_NAME)

