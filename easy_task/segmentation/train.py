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
net = model.fcn2(beta=beta, spike_grad=spike_grad).to(device)


def forward_pass(net, data):
    soft = nn.Softmax2d()
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(data.size(0)):  # data.size(0) = number of time steps
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)

    spk_rec = torch.stack(spk_rec)
    # print(spk_rec.shape)
    spk_cnt = compute_loss.spike_count(spk_rec, channel=True)# batch channel(n_class) pixel pixel 
    spk_cnt_ = spk_cnt[0,0,:,:].reshape(pixel,pixel).to('cpu').detach().numpy().copy()
    # plt.figure()
    # plt.imshow(spk_cnt_)
    # plt.show()
    print(np.sum(spk_cnt_.reshape(-1)))
    # pred_pro = soft(spk_cnt)
    pred_pro = F.softmax(spk_cnt, dim=1)
    pred_pro_ = pred_pro[0,0,:,:].reshape(pixel,pixel).to('cpu').detach().numpy().copy()
    pred_pro__ = pred_pro[0,1,:,:].reshape(pixel,pixel).to('cpu').detach().numpy().copy()
    
    # plt.figure()
    # ax1 = plt.subplot(1,2,1)
    # ax2 = plt.subplot(1,2,2)
    # ax1.imshow(pred_pro_)
    # ax1.set_title('not safe')
    # ax2.imshow(pred_pro__)
    # ax2.set_title('safe')
    # plt.show()
    return pred_pro

optimizer = torch.optim.Adam(net.parameters(), lr=10e-4, betas=(0.9, 0.999))
# loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
weights = torch.tensor([1.0, 3.0]).cuda()
criterion = nn.CrossEntropyLoss(weight=weights)

num_epochs = 30
num_iters = 50
pixel = 64
correct_rate = 0.5
loss_hist = []
hist = defaultdict(list)
# training loop
for epoch in tqdm(range(num_epochs)):
    for i, (data, label) in enumerate(iter(train_loader)):
        data = data.to(device)
        # label = torch.tensor(label, dtype=torch.int64)
        label = label.type(torch.int64)
        label = label.to(device)
        batch = len(data[0])
        data = data.reshape(num_steps, batch, 1, pixel, pixel)
        # print(data.shape)
        net.train()
        pred_pro = forward_pass(net, data)# batch, channel, pixel ,pixel
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
    tqdm.write(f'{acc=}')
    with torch.no_grad():
        net.eval()
        for i, (data, label) in enumerate(iter(test_loader)):
            data = data.to(device)
            label = label.to(device)
            label = label.type(torch.int64)
            batch = len(data[0])
            data = data.reshape(num_steps, batch, 1, pixel, pixel)
            pred_pro = forward_pass(net, data)
            loss_val = criterion(pred_pro, label)
            acc = compute_loss.culc_iou(pred_pro, label, correct_rate)
            hist['test'].append(acc)

## save model
enddir = "models/model1.pth"
torch.save(net.state_dict(), enddir)
print("success model saving")
# Plot Loss
print(hist)
fig = plt.figure(facecolor="w")
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)
ax1.plot(hist['loss'], label="train")
ax1.set_title("Train Set IoU")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("MSE Loss(spike count)")
ax2.plot(hist['train'], label="train")
ax2.set_title("Train Set IoU")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Accuracy")
ax3.plot(hist['test'], label='test')
ax3.set_title("Train Set IoU")
ax3.set_xlabel("epoch")
ax3.set_ylabel("Accuracy")
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