import glob
import scipy.io
import numpy as  np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2
import os
from tqdm import tqdm
import h5py
import shutil
import time
#

from DEM.ransac import *
from module.const import *
from module import convert_label, cmd
from module.const import *
# from sklearn.preprocessing import MinMaxScaler




if __name__ == '__main__':
    bool_only_boulder = False
    if bool_only_boulder:
        if os.path.exists(RAW_EVENT_ONLY_BOULDER_PATH):
            shutil.rmtree(RAW_EVENT_ONLY_BOULDER_PATH)
        os.makedirs(RAW_EVENT_ONLY_BOULDER_PATH)


        data_num = 10
        label_path = LABEL_BOULDER_PATH

        for file_num in tqdm(range( data_num)):
            num_zip = str(file_num).zfill(5)
            label_path = os.path.join(LABEL_BOULDER_PATH, f"{num_zip}.npy")
            label = np.load(label_path)
            # plt.figure()
            # plt.imshow(label)
            # plt.show()
            


            savefile_path = f'{str(file_num).zfill(5)}.h5'
            savefile_path = os.path.join(RAW_EVENT_ONLY_BOULDER_PATH, savefile_path)
            with h5py.File(savefile_path, 'a') as f:
                f.create_dataset("label", data=label)
            
            cmd.v2e_cmd(file_num, EVENT_TH, RAW_EVENT_ONLY_BOULDER_PATH, VIDEO_ONLY_BOULDER_PATH)
            

    else:
        if os.path.exists(ANN_DATASET_PATH):
            shutil.rmtree(ANN_DATASET_PATH)
        os.makedirs(ANN_DATASET_PATH)


        data_num = 3000
        label_path = LABEL_PATH

        for file_num in tqdm(range( data_num)):
            num_zip = str(file_num).zfill(5)
            label_path = os.path.join(LABEL_PATH, f"{num_zip}.npy")
            label = np.load(label_path)
            label = cv2.resize(label, (INPUT_WIDTH, INPUT_HEIGHT))

            video_path = os.path.join(VIDEO_PATH, f"{num_zip}.avi")
            cap = cv2.VideoCapture(video_path)
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # print(frame.shape)
            frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
            # print(frame.shape)
            
            # print(frame.shape)
            # fig = plt.figure()
            # ax1 = fig.add_subplot(121)
            # ax2 = fig.add_subplot(122)
            # ax1.imshow(frame)
            # ax2.imshow(label)
            # plt.show()
            # print(frame.shape)
            frame = np.expand_dims(frame, 0)
            
            # frame = np.transpose(frame, (2,0,1))
            savefile_path = f'{str(file_num).zfill(5)}.h5'
            savefile_path = os.path.join(ANN_DATASET_PATH, savefile_path)
            with h5py.File(savefile_path, 'w') as f:
                f.create_dataset("label", data=label)
                f.create_dataset("input", data=frame)
                
            
 
            

        
        

        
