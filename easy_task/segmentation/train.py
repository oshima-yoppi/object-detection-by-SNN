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

from custom_data import LoadDataset
import custom_data
from model import model, compute_loss

import matplotlib.pyplot as plt
from IPython.display import HTML

from collections import defaultdict
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_data.custom_collate, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=custom_data.custom_collate, shuffle=False,)


def print_batch_accuracy(data, label, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((label == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")





spike_grad = surrogate.atan()
net = model.cnn(beta=beta, spike_grad=spike_grad).to(device)


def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(data.size(0)):  # data.size(0) = number of time steps
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))
# loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

num_epochs = 100
num_iters = 50
pixel = 64
correct_rate = 0.8
loss_hist = []
acc_hist = defaultdict(list)
# training loop
for epoch in tqdm(range(num_epochs)):
    for i, (data, label) in enumerate(iter(train_loader)):
        data = data.to(device)
        label = label.to(device)
        batch = len(data[0])
        data = data.reshape(num_steps, batch, 1, pixel, pixel)

        net.train()
        spk_rec = forward_pass(net, data)# time batch neuron ???
        loss_val = compute_loss.spike_mse_loss(spk_rec, label, rate=correct_rate)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # print(f"Epoch {epoch}, Iteration {i} /nTrain Loss: {loss_val.item():.2f}")

        acc = SF.accuracy_rate(spk_rec, label)
        acc_hist['train'].append(acc)
        # print(f"Accuracy: {acc * 100:.2f}%/n")
        # break

    with torch.no_grad():
        net.eval()
        for i, (data, label) in enumerate(iter(test_loader)):
            data = data.to(device)
            label = label.to(device)
            batch = len(data[0])
            data = data.reshape(num_steps, batch, 1, pixel, pixel)
            spk_rec = forward_pass(net, data)
            loss_val = compute_loss.spike_mse_loss(spk_rec, label)
            acc = SF.accuracy_rate(spk_rec, label)
            acc_hist['test'].append(acc)
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


## save model
enddir = "models/model1.pth"
torch.save(net.state_dict(), enddir)
print("success model saving")

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