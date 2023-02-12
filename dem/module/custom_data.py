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
import numpy as np
import tonic
import tonic.transforms as transforms
from torchvision import transforms as transforms_
import torchvision.transforms as T
from tqdm import tqdm
# from .module import const
from .const import *
# import const

class num2torch():
    def __init__(self):
        return
    def __call__(self, arr):
        return torch.from_numpy(arr)


def convert_raw_event(events_raw_dir, new_dir, accumulate_time):
    SENSOR_SIZE = (IMG_WIDTH, IMG_HEIGHT, 2) # (WHP)
    # converter = transforms.ToFrame(sensor_size=SENSOR_SIZE, time_window=accumulate_time)
    converter = transforms.Compose(
        [transforms.ToFrame(sensor_size=SENSOR_SIZE, time_window=accumulate_time),
        num2torch(),
        transforms_.Resize(size=(INPUT_HEIGHT, INPUT_WIDTH),interpolation=T.InterpolationMode.NEAREST)]
    )
    converter_label = transforms.Compose(
        [transforms_.ToTensor(),
        transforms_.Resize(size=(INPUT_HEIGHT, INPUT_WIDTH),interpolation=T.InterpolationMode.NEAREST)]
    )
    try:
        os.makedirs(new_dir)
        h5py_allfile = glob.glob(f'{events_raw_dir}/*.h5')
        dtype = [('t', '<i4'), ('x', '<i4'), ('y', '<i4'), ('p', '<i4')]
        # print(h5py_allfile)
        for i, file in enumerate(tqdm(h5py_allfile)):
            with h5py.File(file, "r") as f:
                label = f['label'][()]
                raw_events = f['events'][()]
            raw_event_len = raw_events.shape[0]
            processed_events = np.zeros(raw_event_len, dtype=dtype)
            for idx, (key , _) in enumerate(dtype):
                processed_events[key] = raw_events[:,idx]
            acc_events = converter(processed_events)

            label = converter_label(label)
            file_name = f'{str(i).zfill(5)}.h5'
            new_file_path = os.path.join(new_dir, file_name)
            with h5py.File(new_file_path, "w") as f :
                f.create_dataset('label', data=label)
                f.create_dataset('events', data=acc_events)
    except Exception as e:
        import shutil
        shutil.rmtree(new_dir)
        import traceback
        traceback.print_exc()
        exit()
    return 
class LoadDataset(Dataset):
    def __init__(self, processed_event_dataset_path, raw_event_dir,accumulate_time : int, input_height, input_width, train:bool, ):
        
        self.accumulate_time = accumulate_time
        self.input_height = input_height
        self.input_width = input_width
        
        if os.path.isdir(processed_event_dataset_path):
            pass
        else:
            convert_raw_event(events_raw_dir=raw_event_dir, new_dir=processed_event_dataset_path, accumulate_time=self.accumulate_time)

        self.all_files = glob.glob(f"{processed_event_dataset_path}/*")
        self.divide = int((len(self.all_files)*0.2))

        if train:
            self.file_lst = self.all_files[self.divide:]
        else:
            self.file_lst = self.all_files[:self.divide]
    def __len__(self):
        return len(self.file_lst)

    def __getitem__(self, index):
        with h5py.File(self.file_lst[index], "r") as f:
            label = f['label'][()]
            input = f['events'][()]
        input = torch.from_numpy(input.astype(np.float32)).clone()
        label = torch.from_numpy(label.astype(np.float32)).clone()
        return input, label
def custom_collate(batch):
    """
    batchをどの引数に持っていくかを決める関数。入力はbatchを2つ目。ラベルは1つ目に設定。
    """
    input_lst = []
    target_lst = []
    for input, target in batch:
        input_lst.append(input)
        target_lst.append(target)
    return torch.stack(input_lst, dim=1), torch.stack(target_lst, dim=0)

if __name__ == "__main__":
    print(os.getcwd())
    dataset_path = "dataset/"
    raw_event_dir = 'data'
    # youtube_path = "gomibako/h5.gif"
    a = LoadDataset(dir=dataset_path, raw_event_dir=raw_event_dir,accumulate_time=100000 ,train=True)
    input, label = a[0]
    print(input.shape)
    print(label.shape)
    print(input.shape[0])
    # print(a.file_lst[6], a.num_lst[6])
    # make_data.youtube(input, youtube_path)
    
