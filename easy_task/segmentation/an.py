# import torch
# import torch.nn as nn
# c = nn.MSELoss()
# a = torch.zeros((2,3))
# b = torch.ones((2,3))
# print(b * 5)

import torch
a = torch.tensor([1, 0,0])
b =  torch.tensor([1, 1,0])
print(torch.logical_or(a,b))
