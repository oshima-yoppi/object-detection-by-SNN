import cv2
import torch 
import numpy as np
from .const import *

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    return frame

def get_first_events(events):
    events = events.to('cpu')
    if BOOL_DISTINGUISH_EVENT:
        first_events = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3)) # bgr
        first_events[:,:,0] = events[0,0,0] * 255# r 
        first_events[:,:,1] = events[0,0,1] * 255
        return first_events