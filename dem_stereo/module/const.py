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
import json
import numpy as np
import random
from . import rand

rand.give_seed()
with open("module/const_base.json") as file:
    constants = json.load(file)
# 設定ファイル。ここでいろんな変数を定義


soft_reset = constants["soft_reset"]
FINISH_STEP = constants["FINISH_STEP"]
ACCUMULATE_EVENT_MILITIME = constants["ACCUMULATE_EVENT_MILITIME"]
EVENT_COUNT = constants["EVENT_COUNT"]
EVENT_TH = constants["EVENT_TH"]
TIME_CHANGE = constants["TIME_CHANGE"]
BETA_LEARN = constants["BETA_LEARN"]
THRESHOLD_LEARN = constants["THRESHOLD_LEARN"]
REPEAT_INPUT = constants["REPEAT_INPUT"]
BETA = constants["BETA"]
THRESHOLD = constants["THRESHOLD"]
TIME_AWARE_LOSS = constants["TIME_AWARE_LOSS"]
print(
    f"soft_reset: {soft_reset}",
    # f"parm_learn: {PARM_LEARN}",
    f"FINISH_STEP: {FINISH_STEP}",
)
# イベントかめらの極性を分けるかどうか
BOOL_DISTINGUISH_EVENT = True
INPUT_CHANNEL = 2 if BOOL_DISTINGUISH_EVENT else 1

# network関連の定数
INPUT_HEIGHT, INPUT_WIDTH = 130, 173
NETWORK_INPUT_HEIGHT, NETWORK_INPUT_WIDTH = 130, 164
SPLITED_INPUT_HEIGHT, SPLITED_INPUT_WIDTH = INPUT_HEIGHT // 3, INPUT_WIDTH // 3
ROUGH_PIXEL = 3
RIGHT_IDX = 91
LEFT_IDX = INPUT_WIDTH - RIGHT_IDX
# INPUT_HEIGHT, INPUT_WIDTH = 65, 86
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 36
BATCH_SIZE_TEST = 1
# FINISH_STEP = 8 # 8

if soft_reset:
    RESET = "subtract"
else:
    RESET = "zero"
MAX_STEP = 10
if TIME_CHANGE:
    START_STEP = FINISH_STEP + 2
else:
    START_STEP = FINISH_STEP
if START_STEP > MAX_STEP:
    print("error!!!\n\nSTART_STEP > MAX_STEP")
# if TIME_CHANGE:
#     START_STEP = FINISH_STEP + 2
# else:
#     START_STEP = 8

SPIKE_GRAD = surrogate.atan(alpha=1.5)
LR = 1e-4
CORRECT_RATE = 0.5
LOSS_RATE = 1e-7

NET = network.RoughConv3(
    beta=BETA,
    threshold=THRESHOLD,
    spike_grad=SPIKE_GRAD,
    device=DEVICE,
    input_height=NETWORK_INPUT_HEIGHT,
    input_width=NETWORK_INPUT_WIDTH,
    rough_pixel=3,
    beta_learn=BETA_LEARN,
    threshold_learn=THRESHOLD_LEARN,
    input_channel=INPUT_CHANNEL,
    power=True,
    reset=RESET,
    repeat_input=REPEAT_INPUT,
    time_aware_loss=TIME_AWARE_LOSS,
)


LABEL_PATH = "blender/label/"
# LABEL_PATH = "blender/"
LABEL_BOULDER_PATH = "label_only_boulder"

# BOOL_LEARGE_DATASET = False
BOOL_LEARGE_DATASET = True

# ACCUMULATE_EVENT_MILITIME = 100 #[ms] # 何msイベントをためるか

ACCUMULATE_EVENT_MICROTIME = ACCUMULATE_EVENT_MILITIME * 1000  # [us]
DATASET_PATH = "dataset"  # datasetのパス
# DATASET_ACCEVENT_PATH = os.path.join(DATASET_PATH, str(ACCUMULATE_EVENT_MICROTIME)) # dataset/〇〇  ←何秒ためるかを表す

RAW_EVENT_PATH = f"raw-data/th-{str(EVENT_TH)}"  # v2eから出力されたイベント生データ
RAW_EVENT_LEFT_PATH = f"raw-data/th-{str(EVENT_TH)}/left"
RAW_EVENT_RIGHT_PATH = f"raw-data/th-{str(EVENT_TH)}/right"
RAW_EVENT_DIR = f"raw-data/th-{str(EVENT_TH)}/"
PROCESSED_EVENT_DATASET_PATH = f"dataset/{ACCUMULATE_EVENT_MICROTIME}_({INPUT_HEIGHT},{INPUT_WIDTH})_th-{EVENT_TH}_MaxStep-{MAX_STEP}_EventCount-{EVENT_COUNT}_Distinguish-{BOOL_DISTINGUISH_EVENT}_LeargeData-{BOOL_LEARGE_DATASET}"


NETWORK_CLASS_NAME = NET.__class__.__name__

MODEL_NAME = f"{NETWORK_CLASS_NAME}_{ACCUMULATE_EVENT_MICROTIME}_th-{EVENT_TH}_beta-{BETA}_thresh-{THRESHOLD}_BetaLearn-{BETA_LEARN}_thLearn-{THRESHOLD_LEARN}_RepeatInput-{REPEAT_INPUT}_TimeAwareLoss-{TIME_AWARE_LOSS}_TimeChange-{TIME_CHANGE}_step-({START_STEP},{FINISH_STEP})"

MODEL_PATH = f"models/{MODEL_NAME}.pth"

RESULT_PATH = f"result_img/{MODEL_NAME}"


RESULT_ONLY_BOULDER_PATH = f"result_img_boulder/{MODEL_NAME}"


# print(NETWORK_CLASS_NAME)
