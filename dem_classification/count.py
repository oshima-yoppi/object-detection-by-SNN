import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as SF
from snntorch import surrogate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tonic import DiskCachedDataset
import tonic

import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm

from module.custom_data import LoadDataset
from module import custom_data, compute_loss, network
from module.const import *
import matplotlib.pyplot as plt
from IPython.display import HTML

from collections import defaultdict

import yaml

# def print_batch_accuracy(data, label, train=False):
#     output, _ = net(data.view(BATCH_SIZE, -1))
#     _, idx = output.sum(dim=0).max(1)
#     acc = np.mean((label == idx).detach().cpu().numpy())

#     if train:
#         print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
#     else:
#         print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

# Network Architecture
# num_inputs = 28*28
# num_hidden = 1000
# num_outputs = 10
# dtype = torch.float
def main():
    train_dataset = LoadDataset(
        processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH,
        raw_event_dir=RAW_EVENT_PATH,
        accumulate_time=ACCUMULATE_EVENT_MICROTIME,
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH,
        train=True,
        finish_step=FINISH_STEP,
    )
    test_dataset = LoadDataset(
        processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH,
        raw_event_dir=RAW_EVENT_PATH,
        accumulate_time=ACCUMULATE_EVENT_MICROTIME,
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH,
        train=False,
        finish_step=FINISH_STEP,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=custom_data.custom_collate,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=custom_data.custom_collate,
        shuffle=False,
    )

    counts = defaultdict(int)
    for i, (data, label) in enumerate(iter(train_loader)):
        label_class = label.argmax(dim=1)
        for i in label_class:
            counts[i.item()] += 1
    print(counts)  # defaultdict(<class 'int'>, {0: 977, 1: 1423})

    # return


if __name__ == "__main__":
    main()
