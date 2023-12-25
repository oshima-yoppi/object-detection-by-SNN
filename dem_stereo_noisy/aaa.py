import numpy as np
import matplotlib.pyplot as plt


n = 0
path = f"blender/dem/{str(n).zfill(5)}.npy"
data = np.load(path)
print(data.shape)
# data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
# plt.imshow(data)
# plt.show()
