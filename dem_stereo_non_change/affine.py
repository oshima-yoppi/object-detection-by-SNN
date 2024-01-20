# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

lif = snn.Leaky(
    beta=0.8,
    threshold=1.0,
    # reset_mechanism="subtract",
    # reset_mechanism="zero",
    reset_mechanism="none",
)

num_steps = 50
# Initialize membrane, input, and output
mem = torch.ones(1) * 0.9  # U=0.9 at t=0
mem = torch.where(torch.rand(num_steps) < 0.5, 1, 0)  # randomize U
print(mem)
cur_in = torch.zeros(num_steps)  # I=0 for all t
cur_in = mem
spk_out = torch.zeros(1)  # initialize output spikes
mem[0] = 0
mem_rec = [mem]
spk_rec = [spk_out]
# pass updated value of mem and cur_in[step]=0 at every time step
for step in range(num_steps):
    spk_out, mem = lif(cur_in[step], mem)
    print(spk_out.shape, mem.shape)
    # Store recordings of membrane potential
    mem_rec.append(mem)
    # spk_rec.append(spk_out)
# convert the list of tensors into one tensor
mem_rec = torch.stack(mem_rec)
# spk_rec = torch.stack(spk_rec)
print(spk_out)
plt.subplot(3, 1, 1)
plt.plot(mem.numpy())
plt.subplot(3, 1, 2)
plt.plot(mem_rec.numpy())
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(spk_rec)
plt.show()
