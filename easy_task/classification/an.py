import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




fig, axes = plt.subplots(nrows=1, ncols=2)

for ax in axes:
    x = np.random.rand(4,4)
    y = np.random.rand(4,4)
    r = np.random.rand(4,4)
    im = ax.scatter(x, y, s=300, c=r, cmap='jet')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

plt.show()