from module.const import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

# from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import sys
from tqdm import tqdm
import pandas as pd
import argparse
import h5py
import glob
import cv2
import numpy as np
import tonic
import tonic.transforms as transforms
from torchvision import transforms as transforms_
import torchvision.transforms as T
from tqdm import tqdm
import shutil
import pickle

# INPUT_HEIGHT, INPUT_WIDTH = 130, 173
INPUT_HEIGHT, INPUT_WIDTH = 50, 50


def make_dataset_for_akida(
    akida_dataset_dir, events_raw_dir, accumulate_time, finish_step=1, count=False
):
    SENSOR_SIZE = (IMG_WIDTH, IMG_HEIGHT, 2)  # (WHP)
    input_lst = []
    label_lst = []

    import shutil

    if os.path.exists(akida_dataset_dir) == False:
        # shutil.rmtree(akida_dataset_dir)
        os.makedirs(akida_dataset_dir)
    h5py_allfile = glob.glob(f"{events_raw_dir}/*.h5")

    for i, file in enumerate(tqdm(h5py_allfile)):
        with h5py.File(file, "r") as f:
            label = f["label"][()]
            raw_events = f["events"][()]
        input = np.zeros_like(label)
        # print(input.shape)
        for time, y, x, p in raw_events:
            # print(x,y )
            input[x, y] += 1
        # print(raw_events.shape)
        # print(label.shape)
        if count:
            input = cv2.resize(
                input, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR
            )
            label = cv2.resize(
                label, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST
            )
        else:
            input = cv2.resize(
                input, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST
            )
            label = cv2.resize(
                label, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST
            )

        # print(input.shape)
        # break
        if count == False:
            input = np.where(input >= 1, 1, 0)

        input_lst.append(input)
        label_lst.append(label)
        # break
    save_path = os.path.join(
        akida_dataset_dir, f"dataset_{INPUT_WIDTH}_{INPUT_HEIGHT}_count-{count}.pickle"
    )
    with open(save_path, mode="wb") as f:
        pickle.dump((input_lst, label_lst), f)

    return


def make_imgdataset_for_akida(akida_dataset_dir, img_raw_dir):
    SENSOR_SIZE = (IMG_WIDTH, IMG_HEIGHT, 2)  # (WHP)
    input_lst = []
    label_lst = []

    import shutil

    if os.path.exists(akida_dataset_dir) == False:
        # shutil.rmtree(akida_dataset_dir)
        os.makedirs(akida_dataset_dir)
    h5py_allfile = glob.glob(f"{img_raw_dir}/*.h5")

    for i, file in enumerate(tqdm(h5py_allfile)):
        with h5py.File(file, "r") as f:
            # print(f.keys())
            label = f["label"][()]
            input = f["input"][()]
        input = cv2.resize(
            input, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR
        )
        label = cv2.resize(
            label, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST
        )

        input_lst.append(input)
        label_lst.append(label)
        # break
    save_path = os.path.join(
        akida_dataset_dir, f"dataset_{INPUT_WIDTH}_{INPUT_HEIGHT}_ann.pickle"
    )
    with open(save_path, mode="wb") as f:
        pickle.dump((input_lst, label_lst), f)

    return


if __name__ == "__main__":
    AKIDA_DATASET_DIR = "akida"
    img_raw_dir = "dataset_ann"
    event_dir = RAW_EVENT_PATH
    count = True
    make_dataset_for_akida(
        akida_dataset_dir=AKIDA_DATASET_DIR,
        events_raw_dir=RAW_EVENT_PATH,
        accumulate_time=1e6,
        finish_step=1,
        count=count,
    )

    # make_imgdataset_for_akida(akida_dataset_dir=AKIDA_DATASET_DIR, img_raw_dir=img_raw_dir)
