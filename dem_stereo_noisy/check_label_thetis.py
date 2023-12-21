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
from module import view
from module.const import *
from module.custom_data import LoadDataset
import matplotlib.pyplot as plt


def youtube(events, path, bool_split):
    images = []
    time = events.shape[0]
    x = events.shape[2]
    y = events.shape[3]

    if bool_split:
        img_arr = torch.zeros(time, 3, x, y)
        img_arr[:, 0] = events[:, 0]
        img_arr[:, 1] = events[:, 1]
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
    check_save_dir = "check/h5.gif"
    video_center_dir = VIDEO_CENTER_PATH
    label_center_dir = LABEL_CENTER_PATH
    # a = LoadDataset(
    #     processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH,
    #     raw_event_dir=RAW_EVENT_PATH,
    #     accumulate_time=ACCUMULATE_EVENT_MICROTIME,
    #     input_height=INPUT_HEIGHT,
    #     input_width=INPUT_WIDTH,
    #     train=False,
    #     finish_step=FINISH_STEP,
    # )
    custom_number = input("Do u want to see spesific nunber? [y/n]")
    number = -1
    while 1:
        if custom_number == "y":
            number = int(input("何番を読み込む？"))
        else:
            number += 1
        number = str(number).zfill(5)
        video_path = os.path.join(video_center_dir, number + ".avi")
        label_path = os.path.join(label_center_dir, number + ".npy")
        first_frame = view.get_first_frame(video_path)
        first_events = np.load(label_path)
        plt.subplot(1, 2, 1)
        plt.imshow(first_frame)
        plt.subplot(1, 2, 2)

        plt.imshow(first_events)
        plt.show()
