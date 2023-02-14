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

def youtube(events, path, bool_split):
    
    images = []
    time = events.shape[0]
    x = events.shape[2]
    y = events.shape[3]

    if bool_split:
        img_arr = torch.zeros(time, 3, x, y)
        img_arr[:,0] = events[:,0]
        img_arr[:,1] = events[:,1]
        for i in range(time):
            
            p_ = torchvision.transforms.functional.to_pil_image(img_arr[i])
            images.append(p_)
        images[0].save(path, duration = 100, save_all=True, append_images=images[1:], loop = 50)   
    else:
    
        events = torch.logical_or(events[:,0,:,:], events[:,1,:,:]).float()
        for i in range(time):
            # p_ = Image.fromarray(events[i, :,:])
        
            p_ = torchvision.transforms.functional.to_pil_image(events[i,:,:])
            images.append(p_)
        images[0].save(path, duration = 100, save_all=True, append_images=images[1:], loop = 50)




if __name__ == "__main__":
    youtube_path = "gomibako/h5.gif"
    a= LoadDataset(processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH, raw_event_dir=RAW_EVENT_PATH, accumulate_time=ACCUMULATE_EVENT_MICROTIME , input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, train=False)
    while 1:

        number = int(input('何番を読み込む？'))
        events, label = a[number]
        print(events.shape)
        youtube(events, youtube_path, True)
        print('save sucess')

    # a= LoadDataset(processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH, raw_event_dir=RAW_EVENT_PATH, accumulate_time=ACCUMULATE_EVENT_MICROTIME , input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, train=True)
    # # number = int(input('何番を読み込む？'))
    # for idx, (events, label) in enumerate(iter(a)):
    #     if events.shape[0] == 9:
    #         pass
    #     else:
    #         print(a.file_lst[idx])
    #         print(events.shape[0] == 9, idx)
    #         break
    

    # print(events.shape)

    # youtube(events, youtube_path, True)
        
