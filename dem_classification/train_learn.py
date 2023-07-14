import torch 
import torch.nn as nn
import torch.optim as optim

import numpy
import torch
import snntorch as snn
from snntorch import utils
x = torch.tensor([[1.]])
optimizer = optim.SGD([x], lr=0.1)
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 1),
            snn.Leaky(beta=0.99,init_hidden=True ),
        )
        # self.fc1 = nn.Linear(1, 1)
        # self.snn = snn.Leaky(beta=0.99,init_hidden=True )
    def forward(self, x):
        # print(1)
        utils.reset(self.net)
        # print(222)
        y = self.net(x)
        
        return y 
model = SNN()
target = torch.tensor([[0.111]])
for epoch in range(100):
    optimizer.zero_grad()
    y = model(x)
    loss = (target-y)**2
    loss.backward()
    optimizer.step()
    print(loss.item(),)
    print(model.net[0].weight.grad)
