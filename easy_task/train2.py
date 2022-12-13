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

from data import LoadDataset

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
batch_size = 8

train_dataset = LoadDataset(dir = dataset_path, train=True)
test_dataset = LoadDataset(dir = dataset_path,  train=False)
# print(train_dataset[data_id][0]) #(784, 100) 
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=False,)


# # Define Network
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # Initialize layers
#         self.fc1 = nn.Linear(num_inputs, num_hidden)
#         self.lif1 = snn.Leaky(beta=beta)
#         self.fc2 = nn.Linear(num_hidden, num_outputs)
#         self.lif2 = snn.Leaky(beta=beta)

#     def forward(self, x):

#         # Initialize hidden states at t=0
#         mem1 = self.lif1.init_leaky()
#         mem2 = self.lif2.init_leaky()

#         # Record the final layer
#         spk2_rec = []
#         mem2_rec = []

#         for step in range(num_steps):
#             cur1 = self.fc1(x)
#             spk1, mem1 = self.lif1(cur1, mem1)
#             cur2 = self.fc2(spk1)
#             spk2, mem2 = self.lif2(cur2, mem2)
#             spk2_rec.append(spk2)
#             mem2_rec.append(mem2)

#         return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
def print_batch_accuracy(data, label, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((label == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")






spike_grad = surrogate.atan()
net = nn.Sequential(nn.Conv2d(1, 12, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 32, 5),
                    nn.MaxPool2d(2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(5408, 2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)


def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(data.size(0)):  # data.size(0) = number of time steps
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)

optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

num_epochs = 50
num_iters = 50
pixel = 64
loss_hist = []
acc_hist = defaultdict(list)

# training loop
for epoch in range(num_epochs):
    for i, (data, label) in enumerate(iter(train_loader)):
        data = data.to(device)
        label = label.to(device)
        batch = len(data[0])
        data = data.reshape(num_steps, batch, 1, pixel, pixel)

        net.train()
        spk_rec = forward_pass(net, data)
        loss_val = loss_fn(spk_rec, label)

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

    with torch.no_grad():
        net.eval()
        for i, (data, label) in enumerate(iter(test_loader)):
            data = data.to(device)
            label = label.to(device)
            batch = len(data[0])
            data = data.reshape(num_steps, batch, 1, pixel, pixel)
            net.train()
            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, label)
            acc = SF.accuracy_rate(spk_rec, label)
            acc_hist['test'].append(acc)
# Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(acc_hist['train'], label="train")
plt.plot(acc_hist['test'], label='test')
plt.title("Train Set Accuracy")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()




idx = 0

fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
labels=['0', '1']
print(f"The target label is: {label[idx]}")
plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/oosim/Downloads/ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl/bin'
#  Plot spike count histogram
anim = splt.spike_count(spk_rec[:, idx].detach().cpu(), fig, ax, labels=labels,
                        animate=True, interpolate=1)

HTML(anim.to_html5_video())
anim.save("spike_bar.mp4")