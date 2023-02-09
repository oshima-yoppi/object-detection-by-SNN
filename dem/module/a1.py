from torchvision import transforms as transforms
import torchvision.transforms as T
from tqdm import tqdm
import torch
import numpy as np


a = np.zeros((30,30))
convert = transforms.ToTensor()
b = convert(a)
print(b.shape) # torch.Size([1, 30, 30])