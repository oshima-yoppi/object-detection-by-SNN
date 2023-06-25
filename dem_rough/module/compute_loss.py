from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DiceLoss(nn.Module):
    def __init__(self,):
        super(DiceLoss, self).__init__()
    def forward(self, pred_pro, target):
        batch = len(target)
        # print(target.shape)# torch.Size([12, 1, 100, 100])
        smooth = 1e-5
        # print(pred_pro.shape) #batch, channel , pixel, pixel
        pred_pro = pred_pro[:, 1, :, :]
        pred_pro = pred_pro.reshape(batch, -1)
        target = target.reshape(batch, -1)
        # self.pred_binary = torch.where(pred_pro>=rate, 1, 0)
        # union  = torch.logical_or(self.pred_binary, target).sum(dim=1)
        intersection = (pred_pro * target)
        dice = (2. * intersection.sum(1) + smooth) / (pred_pro.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / batch
        # iou = torch.mean(intersection/(union+eps))
        return dice
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean', weight=self.weight)
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE
class Analyzer():
    def __init__(self, binary_rate=0.5, smooth=1e-5):
        self.smooth = smooth
        self.binary_rate = binary_rate
    def _get_BinaryMap(self, pred_pro, target):
        self.pred_binary = torch.where(pred_pro[:,1]>=self.binary_rate, 1, 0)
        self.pred_binary = self.pred_binary.reshape(-1)
        self.target = target.reshape(-1)
        return
    def get_iou(self, ):
    
        union  = torch.logical_or(self.pred_binary, self.target).sum()
        intersection = torch.logical_and(self.pred_binary, self.target).sum()
        eps = 1e-6
        iou = torch.mean(intersection/(union+eps))
        return iou.item()
    def get_precsion(self,):
        
        intersection = torch.logical_and(self.pred_binary, self.target).sum()
        eps = 1e-6
        prec = torch.mean(intersection/(self.pred_binary.sum()+eps))
        return prec.item()
    
    def get_recall(self,):
        intersection = torch.logical_and(self.pred_binary, self.target).sum()
        eps = 1e-6
        recall = torch.mean(intersection/(self.target.sum()+eps))
        return recall.item()

    def __call__(self, pred_pro, target):
        self._get_BinaryMap(pred_pro, target)
        iou = self.get_iou()
        prec = self.get_precsion()
        recall = self.get_recall()
        return iou, prec,  recall
def culc_iou(pred_pro, target, rate=0.8):
    
    batch = len(target)
    # print(pred_pro.shape) #batch, channel , pixel, pixel
    pred_pro = pred_pro[:, 1, :, :]
    pred_pro = pred_pro.reshape(batch, -1)
    target = target.reshape(batch, -1)
    self.pred_binary = torch.where(pred_pro>=rate, 1, 0)
    union  = torch.logical_or(self.pred_binary, target).sum(dim=1)
    intersection = torch.logical_and(self.pred_binary, target).sum(dim = 1)
    eps = 1e-6
    iou = torch.mean(intersection/(union+eps))
    return iou.item()


if __name__ == "__main__":
    a = torch.ones((16, 2, 64, 64))
    label = torch.ones((16, 64, 64))
    print(culc_iou(a, label))

