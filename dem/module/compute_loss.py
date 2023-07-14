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
    target_spike_count = num_steps * rate * target
    target_spike_count = torch.round(target_spike_count)
    loss = criterion(input_spike_count, target_spike_count)
    return loss


def spike_count(spk_rec: torch.Tensor, channel=False):
    if channel == False:
        spk_rec = spk_rec.squeeze()
    count = torch.sum(spk_rec, dim=0,)
    return count


def loss_dice(pred_pro, target, rate=0.8):
    # https://qiita.com/4Ui_iUrz1/items/4c0efd9c50e344c66665

    batch = len(target)
    # print(target.shape)# torch.Size([12, 1, 100, 100])
    smooth = 1e-5
    # print(pred_pro.shape) #batch, channel , pixel, pixel
    pred_pro = pred_pro[:, 1, :, :]
    pred_pro = pred_pro.reshape(batch, -1)
    target = target.reshape(batch, -1)
    # pred = torch.where(pred_pro>=rate, 1, 0)
    # union  = torch.logical_or(pred, target).sum(dim=1)
    intersection = pred_pro * target
    dice = (2.0 * intersection.sum(1) + smooth) / (
        pred_pro.sum(1) + target.sum(1) + smooth
    )
    dice = 1 - dice.sum() / batch
    # iou = torch.mean(intersection/(union+eps))
    return dice


# def loss_iou(pred_pro, target, rate=0.8):

#     batch = len(target)
#     # print(pred_pro.shape) #batch, channel , pixel, pixel
#     pred_pro = pred_pro[:, 1, :, :]
#     pred_pro = pred_pro.reshape(batch, -1)
#     target = target.reshape(batch, -1)
#     pred = torch.where(pred_pro>=rate, 1, 0)
#     union  = torch.logical_or(pred, target).sum(dim=1)
#     intersection = torch.logical_and(pred, target).sum(dim = 1)
#     eps = 1e-6
#     iou = torch.mean(intersection/(union+eps))
#     return iou


def culc_iou(pred_pro, target, rate=0.8):

    batch = len(target)
    # print(pred_pro.shape) #batch, channel , pixel, pixel
    pred_pro = pred_pro[:, 1, :, :]
    pred_pro = pred_pro.reshape(batch, -1)
    target = target.reshape(batch, -1)
    pred = torch.where(pred_pro >= rate, 1, 0)
    union = torch.logical_or(pred, target).sum(dim=1)
    intersection = torch.logical_and(pred, target).sum(dim=1)
    eps = 1e-6
    iou = torch.mean(intersection / (union + eps))
    return iou.item()


if __name__ == "__main__":
    a = torch.ones((16, 2, 64, 64))
    label = torch.ones((16, 64, 64))
    print(culc_iou(a, label))
