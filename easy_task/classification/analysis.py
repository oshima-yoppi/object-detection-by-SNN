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

# from IPython.display import HTML
import pandas as pd
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


def write_result(info_csv_path, new_csv_path):
    info_df = pd.read_csv(info_csv_path)
    info_df["result"] = None
    for key in tf_dict.keys():
        for number in tf_dict[key]:
            info_df.at[number, "result"] = key
    info_df.to_csv(new_csv_path)
    return info_df


if __name__ == "__main__":
    # Network Architecture
    num_inputs = 28 * 28
    num_hidden = 1000
    num_outputs = 10
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Temporal Dynamics
    num_steps = 10
    beta = 0.95
    dataset_path = "dataset/"
    batch_size = 16

    train_dataset = LoadDataset(dir=dataset_path, train=True)
    test_dataset = LoadDataset(dir=dataset_path, train=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=tonic.collation.PadTensors(batch_first=False),
        shuffle=False,
    )

    spike_grad = surrogate.atan()
    net = model.cnn(beta=beta, spike_grad=spike_grad).to(device)
    model_path = "models/model1.pth"
    net.load_state_dict(torch.load(model_path))

    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    num_epochs = 100
    num_iters = 50
    pixel = 64
    loss_hist = []
    acc_hist = defaultdict(list)
    wrong_num_lst = []
    tf_dict = defaultdict(list)
    tn_lst, tp_lst = [], []
    fn_lst, fp_lst = [], []

    with torch.no_grad():
        net.eval()
        for i, (data, label) in enumerate(tqdm(iter(test_loader))):
            data_num = test_dataset.num_lst[i]
            data = data.to(device)
            label = label.to(device)
            batch = len(data[0])
            data = data.reshape(num_steps, batch, 1, pixel, pixel)
            spk_rec = forward_pass(net, data)

            loss_val = loss_fn(spk_rec, label)
            acc = SF.accuracy_rate(spk_rec, label)
            if acc and label:
                tf_dict["tp"].append(data_num)
            elif acc and label == 0:
                tf_dict["tn"].append(data_num)
            elif label:
                tf_dict["fn"].append(data_num)
            elif label == 0:
                tf_dict["fp"].append(data_num)
            # print(acc)
            acc_hist["test"].append(acc)

    print(tf_dict)
    for key in tf_dict.keys():
        print(key, len(tf_dict[key]))
        print(key, len(tf_dict[key]) / 300)

    idx = 0

    # fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
    fig, ax = plt.subplots(facecolor="w")
    labels = ["0", "1"]
    print(f"The target label is: {label[idx]}")
    anim = splt.spike_count(
        spk_rec[:, idx].detach().cpu(),
        fig,
        ax,
        labels=labels,
        animate=True,
        interpolate=10,
    )

    # HTML(anim.to_html5_video())
    # plt.show()
    anim.save("spike_bar.gif", writer="pillow")

    info_csv_path = "info.csv"
    new_csv_path = "result/result_info.csv"
    new_df = write_result(info_csv_path, new_csv_path)
    print(new_df.head())

    pixel = 64
    # https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for i, ax in enumerate(axes):
        if i == 0:
            key = "tp"
            title = "True Positive"
        elif i == 1:
            key = "fn"
            title = "False Negative"
        x, y, r = [], [], []
        for number in tf_dict[key]:
            x.append(new_df.at[number, "x"])
            y.append(new_df.at[number, "y"])
            r.append(new_df.at[number, "radius"])
        display_radius = list(map(lambda x: x * 2, r))
        im = ax.scatter(x, y, s=display_radius, c=r, cmap="jet")

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, pixel)
        ax.set_ylim(0, pixel)
    fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im,)
    # fig.colorbar(im, cax=cbar_ax)
    fig.tight_layout()
    plt.show()
    # fig, axes = plt.subplots(nrows=1, ncols=2)
    # for i, ax in enumerate(axes):
    #     if i == 0:
    #         key = 'tp'
    #         title = 'True Positive'
    #     elif i == 1:
    #         key = 'fn'
    #         title = 'False Negative'
    #     x, y, r = [],[],[]
    #     for number in tf_dict[key]:
    #         x.append(new_df.at[number, 'x'])
    #         y.append(new_df.at[number, 'y'])
    #         r.append(new_df.at[number, 'radius'])
    #         ax.scatter(x, y, s=r, c=r, cmap='jet')

    #     # plt.colorbar(ax=ax)
    #     ax.set_title(title)
    #     ax.set_xlabel("x")
    #     ax.set_ylabel("y")
    #     ax.set_xlim(0, pixel)
    #     ax.set_ylim(0, pixel)
    # fig.tight_layout()
    # plt.show()
