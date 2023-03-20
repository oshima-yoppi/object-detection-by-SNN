# from module.const import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
import os 
from tqdm import tqdm
import sys
from tqdm import tqdm
import pandas as pd
import argparse
import h5py
import glob
import cv2
import numpy as np
import tonic
import tonic.transforms as transforms
from torchvision import transforms as transforms_
import torchvision.transforms as T
from tqdm import tqdm
import shutil
import pickle

INPUT_HEIGHT, INPUT_WIDTH = 64, 64
def make_dataset_for_akida(akida_dataset_dir, events_raw_dir, ):
    
    input_lst = []
    label_lst = []
    try:
        import shutil
        if os.path.exists(akida_dataset_dir):
            shutil.rmtree(akida_dataset_dir)
        os.makedirs(akida_dataset_dir)
        h5py_allfile = glob.glob(f'{events_raw_dir}/*.h5')

        for i, file in enumerate(tqdm(h5py_allfile)):
            with h5py.File(file, "r") as f:
                label = f['label'][()]
                raw_events = f['input'][()]
            input = np.zeros((INPUT_HEIGHT, INPUT_WIDTH))
            # print(input.shape)
            # print(raw_events.shape)
            input = np.sum(raw_events,axis=0)
            input = input.reshape(INPUT_HEIGHT, INPUT_WIDTH, 1)
            # print(raw_events.shape)
            # print(label.shape)

            # input = cv2.resize(input, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
            # label = cv2.resize(label, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_NEAREST)
            # print(input.shape)
            # break
            # print(label)
            # plt.figure()
            # plt.imshow(input)
            # plt.show()
            
            input_lst.append(input)
            label_lst.append(label)
            # break
        save_path = os.path.join(akida_dataset_dir, f"dataset_{INPUT_WIDTH}_{INPUT_HEIGHT}.pickle")
        with open(save_path, mode="wb") as f:
            pickle.dump((input_lst, label_lst), f)
            
    except Exception as e:
        import shutil
        shutil.rmtree(akida_dataset_dir)
        import traceback
        traceback.print_exc()
        exit()
    return 

if __name__ == "__main__":
    AKIDA_DATASET_DIR = 'akida'
    event_dir = 'dataset'
    make_dataset_for_akida(akida_dataset_dir=AKIDA_DATASET_DIR, events_raw_dir=event_dir)
