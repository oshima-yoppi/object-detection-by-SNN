# import torch
# import torch.nn as nn
# c = nn.MSELoss()
# a = torch.zeros((2,3))
# b = torch.ones((2,3))
# print(b * 5)

import torch
a = torch.tensor([1.234, 2.345, 3.456], dtype=torch.float)
b = torch.floor(a) # 1, 2, 3になる
c = torch.ceil(a)  # 2, 3, 4になる
print(b, type(b), b.dtype)
