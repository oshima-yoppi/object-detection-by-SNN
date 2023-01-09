from tarfile import DIRTYPE
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
import make_data
# from sklearn.model_selection import train_test_split
class LoadDataset(Dataset):
    def __init__(self, train:bool, dir):
        self.dir = dir
        self.all_files = glob.glob(f"{self.dir}/*")
        self.divide = int((len(self.all_files)*0.8))
        
        if train:
            self.file_lst = self.all_files[:self.divide]
        else:
            self.file_lst = self.all_files[self.divide:]
            self.num_lst = [i for i in range(self.divide, len(self.all_files))]
    def __len__(self):
        return len(self.file_lst)

    def __getitem__(self, index):
        with h5py.File(self.file_lst[index], "r") as f:
            label = f['label'][()]
            input = f['input'][()]
        input = torch.from_numpy(input.astype(np.float32)).clone()
        label = torch.from_numpy(label.astype(np.float32)).clone()
        return input, label

if __name__ == "__main__":
    dataset_path = "dataset/"
    youtube_path = "gomibako/h5.gif"
    a = LoadDataset(dir=dataset_path, train=False)
    input, label = a[6]
    print(input)
    print(label)
    print(input.shape[0])
    print(a.file_lst[6], a.num_lst[6])
    make_data.youtube(input, youtube_path)
    
