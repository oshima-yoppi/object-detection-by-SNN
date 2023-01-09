import numpy as np
import cv2
import torch 
import h5py
import pandas as pd
import torchvision
import random
import pandas as pd
import os


def draw_circle(center, radius,time=10, pixel=128):
    img = np.zeros((pixel, pixel, 3), dtype=np.uint8)
    cv2.circle(img, center, radius, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8, shift=0)
    return img
def youtube(events, path):
    images = []
    events = events.to(torch.uint8)
    events *= 255
    for i in range(time):
        p_ = torchvision.transforms.functional.to_pil_image(events[i,:,:])
        images.append(p_)
    images[0].save(path, duration = 100, save_all=True, append_images=images[1:], loop = 50)


def SaveEvents(path, ebents, label):
    with h5py.File(path, 'w') as f:
        f.create_dataset('label', data = label)
        f.create_dataset('input', data = ebents)
    return


if __name__ == "__main__":
    pixel = 64
    time = 10
    noize_rate = 0.1
    number_of_data = 10
    numer_list = []
    radius_list, x_list, y_list = [],[],[]
    label_list = []
    path_list = []
    h5py_path = f"dataset/0009.h5"
    youtube_path = f"0.gif"

    with h5py.File(h5py_path, "r") as f:
        label = f['label'][()]
        events = f['input'][()]
  
    events =  torch.from_numpy(events.astype(np.float32)).clone()
    youtube(events, youtube_path)

    
        
