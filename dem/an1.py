import torch 
import torch.nn as nn
# import cv2
# cv2.imread()
m = nn.AdaptiveMaxPool2d((5, 7))
input = torch.randn(1, 64, 8, 9)
output = m(input)
outlut = 0

print(output.shape)