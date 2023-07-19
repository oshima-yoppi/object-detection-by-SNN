import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import glob

fiellst = glob.glob("label_only/*.npy")
for i, file in enumerate(fiellst):
    # with h5py.File(file, "r") as f:
    #     label = f['label'][()]
    #     # input = f['events'][()]
    #     # print(input.shape, label.shape, label)
    #     input = torch.from_numpy(input.astype(np.float32)).clone()
    #     label = torch.tensor(label, dtype=torch.float)
    # # print(input.shape, label.shape)
    #     ;abe;
    label = np.load(file)
    h, w = label.shape
    rough_label = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            splited_label = label[
                i * h // 3 : (i + 1) * h // 3, j * w // 3 : (j + 1) * w // 3
            ]
            contain_one = np.any(splited_label == 1)
            if contain_one:
                rough_label[i, j] = 1
            print(contain_one)
            print(label.shape)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(label)
    plt.subplot(1, 2, 2)
    plt.imshow(rough_label)
    plt.show()
