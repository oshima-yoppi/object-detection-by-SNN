import cv2
import torch 
import numpy as np
from .const import *

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    return frame

def get_first_events(events):

    if events.dim() == 5: #[TBCHW]
        events = events.to('cpu')
        if BOOL_DISTINGUISH_EVENT:
            first_events = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3)) # bgr
            if EVENT_COUNT:
                first_events = np.zeros((INPUT_HEIGHT, INPUT_WIDTH))
                # first_events[:,:,0] = events[0,0,0]# r 
                # first_events[:,:,1] = events[0,0,1]
                first_events[:,:] = events[0,0,1]
            else:
                first_events[:,:,0] = events[0,0,0] * 255# r 
                first_events[:,:,1] = events[0,0,1] * 255
            # print(first_events.max(), first_events.min())
            return first_events.astype(int)
    elif events.dim() == 4:#[TCHW]
        events = events.to('cpu')
        if BOOL_DISTINGUISH_EVENT:
            first_events = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3)) # bgr
            if EVENT_COUNT:
                first_events[:,:,0] = events[0,0,0]# r 
                first_events[:,:,1] = events[0,0,1]
            else:
                first_events[:,:,0] = events[0,0,0] * 255# r 
                first_events[:,:,1] = events[0,0,1] * 255
            first_events = first_events.astype(int)
            return first_events



def draw_edge_of_areas(imgs, line_color=(255, 0, 0,)):
    height, width, _ = imgs.shape
    region_width = width // 3
    region_height = height // 3
    # 横の線を描画
    for i in range(1, 3):
        start_point = (0, i * region_height)
        end_point = (width, i * region_height)
        cv2.line(imgs, start_point, end_point, line_color, thickness=2)

# 縦の線を描画
    for i in range(1, 3):
        start_point = (i * region_width, 0)
        end_point = (i * region_width, height)
        cv2.line(imgs, start_point, end_point, line_color, thickness=2)
    return imgs 

