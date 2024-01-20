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


def get_images(events):
    images = []
    all_steps = events.shape[0]
    x = events.shape[2]
    y = events.shape[3]
    for event in events:
        image = torch.zeros(3, x, y)
        image[0] = event[0]
        image[1] = event[1]
        image = torchvision.transforms.functional.to_pil_image(image)
        images.append(image)
    return images


if __name__ == "__main__":
    all_steps = FINISH_STEP
    event_th = EVENT_TH
    thesis_dir = "thesis"
    os.makedirs(thesis_dir, exist_ok=True)
    save_name = os.path.join(thesis_dir, "none_events")

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
        events_frames = get_images(events)
        sum_frame = len(events_frames)
        for i, frame in enumerate(events_frames):
            plt.imshow(frame)
            # メモリなし
            plt.tick_params(
                labelbottom=False, labelleft=False, labelright=False, labeltop=False
            )
            plt.savefig(f"{save_name}_{i}.pdf")
