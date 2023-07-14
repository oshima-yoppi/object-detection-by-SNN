import numpy as np
import cv2
import torch
import h5py
import pandas as pd
import torchvision
import random
import pandas as pd
import tonic
import tonic.transforms as transforms
import os
from PIL import Image
from module.const import *
from module.custom_data import LoadDataset
import time


def youtube(events, path, bool_split, time=None):

    images = []
    if time == None:
        time = events.shape[0]
    x = events.shape[2]
    y = events.shape[3]

    if bool_split:
        img_arr = torch.zeros(time, 3, x, y)
        img_arr[:, 0] = events[:time, 0]
        img_arr[:, 1] = events[:time, 1]
        for i in range(time):

            p_ = torchvision.transforms.functional.to_pil_image(img_arr[i])
            images.append(p_)
        images[0].save(
            path, duration=100, save_all=True, append_images=images[1:], loop=50
        )
    else:

        events = torch.logical_or(events[:, 0, :, :], events[:, 1, :, :]).float()
        for i in range(time):
            # p_ = Image.fromarray(events[i, :,:])

            p_ = torchvision.transforms.functional.to_pil_image(events[i, :, :])
            images.append(p_)
        images[0].save(
            path, duration=100, save_all=True, append_images=images[1:], loop=50
        )


if __name__ == "__main__":
    bool_boulder = False
    all_time = int(input("How many time steps??"))
    if bool_boulder:

        events = [0.1, 0.15, 0.2, 0.5]  # 変な値入れるとぶっ壊れる可能性あり。ちゅいい
        number = int(input("何番を読み込む？"))
        for th in events:

            youtube_path = f"gomibako/{th}.gif"
            pro = f"dataset_boulder/{ACCUMULATE_EVENT_MICROTIME}_({INPUT_HEIGHT},{INPUT_WIDTH})_th-{th}"
            raw_path = f"raw-data_only_boulder/th-{str(th)}"
            a = LoadDataset(
                processed_event_dataset_path=pro,
                raw_event_dir=raw_path,
                accumulate_time=ACCUMULATE_EVENT_MICROTIME,
                input_height=INPUT_HEIGHT,
                input_width=INPUT_WIDTH,
                train=False,
            )

            events, label = a[number]
            print(events.shape)
            youtube(events, youtube_path, True)
            print("save sucess")

    else:
        events = [0.15]  # 変な値入れるとぶっ壊れる可能性あり。ちゅいい
        number = int(input("何番を読み込む？"))
        for th in events:

            youtube_path = f"gomibako/{th}.gif"
            pro = f"dataset/{ACCUMULATE_EVENT_MICROTIME}_({INPUT_HEIGHT},{INPUT_WIDTH})_th-{th}"
            raw_path = f"raw-data/th-{str(th)}"
            a = LoadDataset(
                processed_event_dataset_path=pro,
                raw_event_dir=raw_path,
                accumulate_time=ACCUMULATE_EVENT_MICROTIME,
                input_height=INPUT_HEIGHT,
                input_width=INPUT_WIDTH,
                train=False,
            )

            events, label = a[number]
            print(events.shape)
            youtube(events, youtube_path, True, all_time)
            print("save sucess")
