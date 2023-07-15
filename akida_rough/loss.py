import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # print(inputs.shape)
        inputs = F.softmax(inputs, dim=1)       
        
        #flatten label and prediction tensors
        # print(inputs.shape)
        inputs = inputs[:,1,:,:].reshape(-1)
        targets = targets.reshape(-1)
        # print(inputs.shape, targets.shape)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice



#PyTorch
class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)     
          
        inputs = F.softmax(inputs, dim=1)    
        #flatten label and prediction tensors
        
        inputs = inputs[:,1,:,:].reshape(-1)
        targets = targets.reshape(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU


