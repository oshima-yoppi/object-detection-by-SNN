import torch 
 

a = torch.tensor([1,2,3])
b = a
a = a * 2

print(a)
print(b)