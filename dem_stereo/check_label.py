import numpy as np
import matplotlib.pyplot as plt

n = 1
center_path = "blender/label/center"
left_path = "blender/label/left"
right_path = "blender/label/right"


center_labels = np.load(f"{center_path}/{str(n).zfill(5)}.npy")
left_labels = np.load(f"{left_path}/{str(n).zfill(5)}.npy")
right_labels = np.load(f"{right_path}/{str(n).zfill(5)}.npy")

print(center_labels.shape)
for i in range(10):
    plt.subplot(2, 5, i + 1)
    if i == 0:
        plt.imshow(right_labels[i])
    if i != 0:
        plt.imshow(right_labels[i] - right_labels[i - 1])
plt.show()
