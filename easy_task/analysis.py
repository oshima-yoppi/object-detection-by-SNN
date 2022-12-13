import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as SF
from snntorch import surrogate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tonic import DiskCachedDataset
import tonic

import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm

from data import LoadDataset
import model

import matplotlib.pyplot as plt
from IPython.display import HTML

from collections import defaultdict



def print_batch_accuracy(data, label, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((label == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")
def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(data.size(0)):  # data.size(0) = number of time steps
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)
if __name__ == "__main__":
    # Network Architecture
    num_inputs = 28*28
    num_hidden = 1000
    num_outputs = 10
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Temporal Dynamics
    num_steps = 10
    beta = 0.95
    dataset_path = "dataset/"
    batch_size = 16

    train_dataset = LoadDataset(dir = dataset_path, train=True)
    test_dataset = LoadDataset(dir = dataset_path,  train=False)
    # print(train_dataset[data_id][0]) #(784, 100) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=False,)



    spike_grad = surrogate.atan()
    net = model.cnn(beta=beta, spike_grad=spike_grad).to(device)
    model_path = 'models/model1.pth'
    net.load_state_dict(torch.load(model_path))

    

    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    num_epochs = 100
    num_iters = 50
    pixel = 64
    loss_hist = []
    acc_hist = defaultdict(list)
    wrong_num_lst = []
    tn_lst, tp_lst = [], []
    fn_lst, fp_lst = [], []


    with torch.no_grad():
        net.eval()
        for i, (data, label) in enumerate(iter(test_loader)):
            data_num = test_dataset.num_lst[i]
            data = data.to(device)
            label = label.to(device)
            batch = len(data[0])
            data = data.reshape(num_steps, batch, 1, pixel, pixel)
            spk_rec = forward_pass(net, data)
            
            loss_val = loss_fn(spk_rec, label)
            acc = SF.accuracy_rate(spk_rec, label)
            if acc and label: tp_lst.append(data_num)
            elif acc and label == 0: tn_lst.append(data_num)
            elif label:fn_lst.append(data_num)
            elif label == 0 : fp_lst.append(data_num)
            # print(acc)
            acc_hist['test'].append(acc)
    
    print(tn_lst, tn_lst)
    print(fn_lst, fp_lst)
    # Plot Loss
    fig = plt.figure(facecolor="w")
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(acc_hist['train'], label="train")
    ax1.set_title("Train Set Accuracy")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Accuracy")
    ax2.plot(acc_hist['test'], label='test')
    ax2.set_title("Train Set Accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("Accuracy")
    fig.tight_layout()

    plt.show()

    # idx = 0

    # fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    # labels=['0', '1']
    # print(f"The target label is: {label[idx]}")
    # plt.rcParams['animation.ffmpeg_path'] = r'C:/Users/oosim/Downloads/ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl/bin'
    # #  Plot spike count histogram
    # # print(spk_rec.shape) #torch.Size([time, batch label])
    # anim = splt.spike_count(spk_rec[:, idx].detach().cpu(), fig, ax, labels=labels,
    #                         animate=True, interpolate=1)

    # HTML(anim.to_html5_video())
    # anim.save("spike_bar.mp4")

