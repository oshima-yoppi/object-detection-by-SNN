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

# import IPython
# from IPython.display import Video
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def youtube(events, path, bool_split, all_steps=None):
    images = []
    plt_imgs = []
    plt_imgs0 = []
    plt_imgs1 = []
    plt_imgs2 = []
    animation_frame = []
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    # fig = plt.figure()
    # fig0 = plt.figure()

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
            # plt_imgs.append([plt.imshow(img_arr[i].permute(1, 2, 0))])
            sum = img_arr[i, 0] + img_arr[i, 1]
            print(sum.shape)
            print(torch.max(sum))
            frame1 = ax1.imshow(sum)
            frame2 = ax2.imshow(torch.where(sum >= 2, 1, 0))
            frame3 = ax3.imshow(torch.where(sum > 8, 1, 0))
            animation_frame.append([frame1, frame2, frame3])
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
    # print(events)
    print(torch.max(events))
    # ani = animation.ArtistAnimation(fig, plt_imgs, interval=100)
    # plt.show()
    ani = animation.ArtistAnimation(fig, animation_frame, interval=100)
    plt.show()

    # fig1 = plt.figure()
    # ani1 = animation.ArtistAnimation(fig1, plt_imgs1, interval=100)
    # plt.show()

    # fig2 = plt.figure()
    # ani2 = animation.ArtistAnimation(fig2, plt_imgs2, interval=100)
    # plt.show()
    # ani.save("gomibako/count.gif", writer="pillow")
    # ani0.save("gomibako/sum.gif", writer="pillow")
    # ani1.save("gomibako/sum4.gif", writer="pillow")
    # ani2.save("gomibako/sum8.gif", writer="pillow")


all_steps = FINISH_STEP
event_th = EVENT_TH
youtube_path = f"gomibako/FIG_{event_th}.gif"
os.makedirs("gomibako", exist_ok=True)

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
    number = int(input("number:"))
    events, label = a[number]
    print(events.shape)
    youtube(events, youtube_path, True, all_steps)
    print("save sucess")
