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
from module import custom_data
from module import network, compute_loss, define_net

import matplotlib.pyplot as plt
from IPython.display import HTML

from collections import defaultdict

import yaml


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

net = define_net.net
def print_batch_accuracy(data, label, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((label == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")



optimizer = torch.optim.Adam(net.network.parameters(), lr=10e-4, betas=(0.9, 0.999))
# loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
weights = torch.tensor([1.0, 3.0]).cuda()
criterion = nn.CrossEntropyLoss(weight=weights)

num_epochs = 300
num_iters = 50
pixel = 64
correct_rate = 0.5
loss_hist = []
hist = defaultdict(list)
# training loop
try:

    for epoch in tqdm(range(num_epochs)):
        for i, (data, label) in enumerate(iter(train_loader)):
            data = data.to(device)
            # label = torch.tensor(label, dtype=torch.int64)
            label = label.type(torch.int64)
            label = label.to(device)
            batch = len(data[0])
            data = data.reshape(num_steps, batch, 1, pixel, pixel)
            # print(data.shape)
            net.network.train()
            pred_pro = net(data)# batch, channel, pixel ,pixel
            # print(pred_pro.shape)
            # loss_val = criterion(pred_pro, label)
            loss_val = compute_loss.loss_dice(pred_pro, label, correct_rate)
            # loss_val = 1 - acc

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            hist['loss'].append(loss_val.item())
            acc = compute_loss.culc_iou(pred_pro, label, correct_rate)

            # print(f"Epoch {epoch}, Iteration {i} /nTrain Loss: {loss_val.item():.2f}")

            
            hist['train'].append(acc)

            # print(f"Accuracy: {acc * 100:.2f}%/n")
            # spk_count_batch = (spk_rec==1).sum().item()
            # spk_count_batch /= batch
            # tqdm.write(f'{spk_count_batch}')
            # plt.figure()
            # plt.imshow(pred_pro[0,1,:,:].to('cpu').detach().numpy())
            # plt.show()
        tqdm.write(f'{epoch}:{acc=}')
        with torch.no_grad():
            net.network.eval()
            for i, (data, label) in enumerate(iter(test_loader)):
                data = data.to(device)
                label = label.to(device)
                label = label.type(torch.int64)
                batch = len(data[0])
                data = data.reshape(num_steps, batch, 1, pixel, pixel)
                pred_pro = net(data)
                loss_val = criterion(pred_pro, label)
                acc = compute_loss.culc_iou(pred_pro, label, correct_rate)
                hist['test'].append(acc)
except:pass
## save model
enddir = "models/model1.pth"
torch.save(net.network.state_dict(), enddir)
print("success model saving")
# Plot Loss
# print(hist)
fig = plt.figure(facecolor="w")
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
ax1.plot(hist['loss'], label="train")
ax1.set_title("loss")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Loss (Dice)")
ax2.plot(hist['train'], label="train")
ax2.set_title("Train  IoU")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Accuracy(IoU)")
ax3.plot(hist['test'], label='test')
ax3.set_title("Test IoU")
ax3.set_xlabel("epoch")
ax3.set_ylabel("Accuracy(IoU)")
fig.tight_layout()

plt.show()


