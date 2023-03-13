import torch
import torch.nn as nn
import torch.nn.functional as F


#
class DiceLoss(nn.Modul):
    def __init__(self, weight=None, size_averate=True
                 )