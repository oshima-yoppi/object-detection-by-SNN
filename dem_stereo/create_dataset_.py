import glob
import scipy.io
import numpy as np
import math

import matplotlib.pyplot as plt

# from mpl_toolkits import mplot3d
import cv2
import os
from tqdm import tqdm
import h5py
import shutil
import time

#

# from DEM.ransac import *
from module.const import *
from module.const_blender import *
from module import convert_label
from module import cmd


if __name__ == "__main__":
    # for path in [RAW_EVENT_LEFT_PATH, RAW_EVENT_RIGHT_PATH]:
    #     if os.path.exists(path):
    #         shutil.rmtree(path)
    #     os.makedirs(path)

    data_num = 3000
    label_path = LABEL_PATH

    for file_num in tqdm(range(data_num)):
        num_zip = str(file_num).zfill(5)
        video_left_path = os.path.join(VIDEO_LEFT_PATH, f"{num_zip}.avi")
        video_right_path = os.path.join(VIDEO_RIGHT_PATH, f"{num_zip}.avi")
        label_left_path = os.path.join(LABEL_LEFT_PATH, f"{num_zip}.npy")
        label_right_path = os.path.join(LABEL_RIGHT_PATH, f"{num_zip}.npy")
        raw_event_left_savepath = os.path.join(RAW_EVENT_LEFT_PATH, f"{num_zip}.h5")
        raw_event_right_savepath = os.path.join(RAW_EVENT_RIGHT_PATH, f"{num_zip}.h5")

        label_left = np.load(label_left_path)
        label_right = np.load(label_right_path)
        # 書き換え
        with h5py.File(raw_event_left_savepath, "r+") as f:
            del f["label"]
            f.create_dataset("label", data=label_left)
        with h5py.File(raw_event_right_savepath, "r+") as f:
            del f["label"]
            f.create_dataset("label", data=label_right)

            # for video_dir, raw_event_dir in zip(
            #     [VIDEO_LEFT_PATH, VIDEO_RIGHT_PATH],
            #     [RAW_EVENT_LEFT_PATH, RAW_EVENT_RIGHT_PATH],
            # ):
            #     save_filename = f"{str(file_num).zfill(5)}.h5"
            #     savefile_path = os.path.join(raw_event_dir, save_filename)
            #     # with h5py.File(savefile_path, "a") as f:
            #     #     f.create_dataset("label", data=label)

            #     file_num = str(file_num).zfill(5)
            #     video_path = os.path.join(video_dir, f"{file_num}.avi")

        # cmd.v2e_cmd(
        #     save_path=raw_event_left_savepath,
        #     video_path=video_left_path,
        #     raw_event_dir=RAW_EVENT_LEFT_PATH,
        # )
        # cmd.v2e_cmd(
        #     save_path=raw_event_right_savepath,
        #     video_path=video_right_path,
        #     raw_event_dir=RAW_EVENT_RIGHT_PATH,
        # )
