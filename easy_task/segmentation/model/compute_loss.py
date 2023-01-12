import torch
import torch.nn as nn


def spike_mse_loss(input, target, rate=0.8):
    # input shape:[time, batch, channel, pixel, pixel]?
    # print(target.shape)
    batch = target.shape[0]
    target = target.reshape(batch, -1)
    criterion = nn.MSELoss()
    num_steps = input.shape[0]
    input_spike_count = torch.sum(input, dim=0)
    target_spike_count = num_steps*rate * target
    target_spike_count = torch.round(target_spike_count)
    loss = criterion(input_spike_count, target_spike_count)
    return loss

def spike_count(spk_rec :torch.Tensor, channel=False):
    if channel==False:
        spk_rec = spk_rec.squeeze()
    count = torch.sum(spk_rec, dim =0)
    count = 0
    return count

def spike_sigmoid_loss(input, target, rate=0.8):
    # input shape:[time, batch, channel, pixel, pixel]?
    # print(target.shape)
    batch = target.shape[0]
    target = target.reshape(batch, -1)
    criterion = nn.MSELoss()
    num_steps = input.shape[0]
    input_spike_count = torch.sum(input, dim=0)
    target_spike_count = num_steps*rate * target
    target_spike_count = torch.round(target_spike_count)
    loss = criterion(input_spike_count, target_spike_count)
    return loss
def show_pred(spike_train, rate=0.8):
    num_steps = spike_train.shape[0]
    batch = spike_train.shape[1]
    th_count = round(num_steps*rate)
    spike_train = spike_train.reshape(num_steps, batch, -1)
    spike_count = torch.sum(spike_train, dim=0)
    pred = torch.where(spike_count >= th_count, 1 ,0)
    return pred
def culc_iou(spike_train, target, rate=0.8):
    
    pred = show_pred(spike_train, rate)
    
    batch = target.shape[0]
    target = target.reshape(batch, -1)

    
    union  = torch.logical_or(pred, target)
    intersection = torch.logical_and(pred, target)
    eps = 1e-6
    iou = torch.mean(intersection/(union+eps))
    return iou.item()
    

