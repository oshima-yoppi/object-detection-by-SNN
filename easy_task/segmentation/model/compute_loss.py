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
    count = torch.sum(spk_rec, dim =0, )
    return count





def culc_iou(pred_pro, target, rate=0.8):
    batch = len(target)
    pred_pro = pred_pro[:, 1, :, :]
    pred_pro = pred_pro.reshape(batch, -1)
    target = target.reshape(batch, -1)
    pred = torch.where(pred_pro>=rate, 1, 0)
    union  = torch.logical_or(pred, target).sum(dim=1)
    intersection = torch.logical_and(pred, target).sum(dim = 1)
    eps = 1e-6
    iou = torch.mean(intersection/(union+eps))
    return iou.item()
    

