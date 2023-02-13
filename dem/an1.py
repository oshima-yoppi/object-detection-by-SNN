import torch 

a = torch.zeros((3,3))
b = torch.ones((2,3))
c = torch.zeros((1,3))

a[:2] = a[:2]+b
print(a)