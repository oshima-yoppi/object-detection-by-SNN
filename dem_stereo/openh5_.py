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
import matplotlib.pyplot as plt


def youtube(events, path, bool_split, all_steps=None):
    images = []
    if all_steps == None:
        all_steps = events.shape[0]
    x = events.shape[2]
    y = events.shape[3]

    if bool_split:
        img_arr = torch.zeros(all_steps, 3, x, y)
        img_arr[:, 0] = events[:all_steps, 0]
        img_arr[:, 1] = events[:all_steps, 1]
        for i in range(all_steps):
            p_ = torchvision.transforms.functional.to_pil_image(img_arr[i])
            images.append(p_)
        images[0].save(
            path, duration=100, save_all=True, append_images=images[1:], loop=50
        )
    else:
        events = torch.logical_or(events[:, 0, :, :], events[:, 1, :, :]).float()
        for i in range(all_steps):
            # p_ = Image.fromarray(events[i, :,:])

            p_ = torchvision.transforms.functional.to_pil_image(events[i, :, :])
            images.append(p_)
        images[0].save(
            path, duration=100, save_all=True, append_images=images[1:], loop=50
        )
    # print(p_.size, max(p_))
    print(events)
    print(torch.max(events))


if __name__ == "__main__":
    all_steps = FINISH_STEP
    event_th = EVENT_TH
    youtube_path = f"gomibako/FIG_{event_th}.gif"

    a = LoadDataset(
        processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH,
        raw_event_dir=RAW_EVENT_DIR,
        accumulate_time=ACCUMULATE_EVENT_MICROTIME,
        finish_step=all_steps,
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH,
        train=False,
    )

    while 1:
        number = int(input("何番を読み込む？"))
        events, label = a[number]
        print(events.shape)
        _, _, h, w = events.shape
        events = events.numpy()
        p_evnets = events[0, 1, :, :]
        n_evnets = events[0, 0, :, :]
        imgs = np.zeros((h, w, 3))
        imgs[:, :, 0] = p_evnets * 255
        imgs[:, :, 1] = n_evnets * 255
        imgs = imgs.astype(np.uint8)
        cv2.imwrite("gomibako/FIG.png", cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR))

        # youtube(events, youtube_path, True, all_steps)
        # print("save sucess")
