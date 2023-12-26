import numpy as np
import matplotlib.pyplot as plt

for i in range(3):
    number = str(i).zfill(5)
    plt.subplot(1, 3, 1)
    label = np.load(f"blender/label/left/{number}.npy")
    print(np.count_nonzero(label == 1))
    print(type(label), label.dtype)
    plt.imshow(np.array(label, dtype=np.uint8))
    label = np.load(f"blender/label/right/{number}.npy")
    plt.subplot(1, 3, 2)
    plt.imshow(label)
    label = np.load(f"blender/label/center/{number}.npy")
    plt.subplot(1, 3, 3)
    plt.imshow(label)
    plt.show()


# data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
# plt.imshow(data)
# plt.show()
