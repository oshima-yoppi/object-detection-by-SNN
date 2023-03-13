import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as SF
from snntorch import surrogate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tonic import DiskCachedDataset
import tonic

import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm
import pickle

# from module.custom_data import LoadDataset
# from module import custom_data, compute_loss, network
# from module.const import *
from loss import *
import matplotlib.pyplot as plt
from IPython.display import HTML

from collections import defaultdict

import yaml

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
kernel = 5
padding = kernel//2
network = nn.Sequential(
    nn.Conv2d(1, 16, kernel, padding=padding),
    nn.ReLU(),
    nn.Conv2d(16, 32, kernel, padding=padding),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel, padding=padding),
    nn.ReLU(),
    nn.Conv2d(32, 32, kernel, padding=padding),
    nn.ReLU(),
    
    nn.Conv2d(32, 16, kernel, padding=padding),
    nn.ReLU(),
    nn.Conv2d(16, 2, kernel, padding=padding),
    # nn.Softmax2d()
    
).to(device=device)
class LoadDataset(Dataset):
    def __init__(self, dataset_path,  train:bool, test_rate=0.2):
        # dataset_path = 'dataset.p
        with open(dataset_path, 'rb') as f:
            self.train_lst, self.label_lst = pickle.load(f)
        length = len(self.train_lst)
        devide = int(length*test_rate)
        if train:
            self.train_lst = self.train_lst[:devide]
            self.label_lst = self.label_lst[:devide]
        else:
            self.train_lst = self.train_lst[devide:]
            self.label_lst = self.label_lst[devide:]
        self.train_lst = np.array(self.train_lst)
        self.label_lst = np.array(self.label_lst)
        self.train_lst = torch.from_numpy(self.train_lst).float()
        self.label_lst = torch.from_numpy(self.label_lst).float()
        self.train_lst = self.train_lst/10
    def __len__(self):
        return len(self.train_lst)

    def __getitem__(self, index):
        return self.train_lst[index], self.label_lst[index]

# width, height = 173, 130
width, height = 50, 50
dataset_path = f'dataset_{width}_{height}.pickle'
train_dataset = LoadDataset(dataset_path=dataset_path, train=True)
test_dataset = LoadDataset(dataset_path=dataset_path, train=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False,)


events, _ = train_dataset[0]
num_steps = events.shape[0]
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, betas=(0.9, 0.999))


num_epochs = 100
num_iters = 50
# pixel = 64
correct_rate = 0.5
loss_hist = []
hist = defaultdict(list)

dice_loss = DiceLoss()
iou_eval = IoU()
# training loop
try:
    for epoch in tqdm(range(num_epochs)):
        for i, (data, label) in enumerate(iter(train_loader)):
            data = data.to(device)
            # label = torch.tensor(label, dtype=torch.int64)
            label = label.type(torch.int64)
            label = label.to(device)
            batch = len(data[0])
            data = data.reshape(-1, 1, height, width)
            # print(data.shape)
            network.train()
            # print(data.shape)
            output = network(data)# batch, channel, pixel ,pixel
            # print(output.shape)
            # loss_val = criterion(output, label)
            loss_val = dice_loss(output, label)
            # loss_val = 1 - acc

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            hist['loss'].append(loss_val.item())
            acc = iou_eval(output, label)

            # print(f"Epoch {epoch}, Iteration {i} /nTrain Loss: {loss_val.item():.2f}")

            
            
            hist['train'].append(acc.item())
            
            # plt.figure()
            # plt.imshow(output[0,1,:,:].to('cpu').detach().numpy())
            # plt.show()
        
            if i % 10 == 0:
                with torch.no_grad():
                    network.eval()
                    for _, (data, label) in enumerate(iter(test_loader)):
                        data = data.to(device)
                        label = label.to(device)
                        data = data.reshape(-1, 1, height, width)
                        # label = label.type(torch.int64)
                        batch = len(data[0])
                        # data = data.reshape(num_steps, batch, INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH)
                        output = network(data)
                        # loss_val = criterion(output, label)
                        acc = iou_eval(output, label)
                        hist['test'].append(acc.item())
                        
                        # del data, label, loss_val, output, acc
                tqdm.write(f'{i}:{acc.item()=}')
except Exception as e:
    import traceback
    print('--------error--------')
    traceback.print_exc()
    print('--------error--------')
    pass
    # print(e)
## save model
# enddir = MODEL_PATH
# # if os.path.exists(enddir) == False:
# #     os.makedirs(enddir)
# torch.save(net.state_dict(), enddir)
# print("success model saving")
# print(MODEL_NAME)
# print(f'{acc=}')
# # Plot Loss
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
# fig.suptitle(f"ModelName:{MODEL_NAME}")
fig.tight_layout()
plt.show()


