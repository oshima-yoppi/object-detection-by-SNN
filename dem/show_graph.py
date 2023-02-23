import matplotlib.pyplot as plt
import numpy as np
import math
x = np.arange(-20,20,1)
fig = plt.figure()
ax1 = fig.add_subplot(221)
y = np.where(x>=0, 1, 0)
ax1.plot(x, y)
ax2 = fig.add_subplot(222)
y = np.zeros_like(x)
ax2.plot(x,y)


ax3 = fig.add_subplot(223)
y = np.arctan(x)
ax3.plot(x, y)

ax4 = fig.add_subplot(224)
def arcf(x):
    y = []
    for x in x:
        y.append(1/(1+x)**2)
    return y
y = arcf(x)
ax4.plot(x, y)
# ax1.set_ylim(-1,2)
# ax2.set_ylim(-1,2)
# ax3.set_ylim(-1,2)
# ax4.set_ylim(-1,2)
plt.show()