#%%
from tensorflow import keras
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
# from google.colab.patches import cv2_imshow #cv2_imshow(img)



# %%
# Load MNIST dataset
# INPUT_HEIGHT, INPUT_WIDTH = 50 ,50
INPUT_HEIGHT, INPUT_WIDTH = 130 ,173
COUNT = False
# COUNT = True

# dataset_path = f"dataset/dataset_{INPUT_WIDTH}_{INPUT_HEIGHT}_count-{COUNT}.pickle"
dataset_path = f"dataset_173_130_count-False.pickle"

with open(dataset_path, 'rb') as f:
    train_lst, label_lst = pickle.load(f)
## 
train_lst = train_lst[:]
label_lst = label_lst[:]
train_lst = np.array(train_lst[::-1])
label_lst = np.array(label_lst[::-1])
print(train_lst.shape)
num_train = len(train_lst)
x_train, x_test, y_train, y_test = train_test_split(train_lst, label_lst, shuffle=False, test_size=0.2, random_state=42)
x_train.shape
x_train.max()

#%%
# Reshape x-data
x_train = x_train.reshape(-1, INPUT_HEIGHT, INPUT_WIDTH, 1)
x_test = x_test.reshape(-1, INPUT_HEIGHT, INPUT_WIDTH, 1)

# Set aside raw test data for use with Akida runtime later
raw_x_test = x_test.astype('uint8')
raw_y_test = y_test.astype('uint8')

# Rescale x-data
if COUNT:
    a = 100 # alltime=1000 ms frame v2e = 10 ms
    # a = 30 # alltime=1000 ms frame v2e = 10 ms
    a = 1
    b = 0
else:
    a = 1
    b = 0
input_scaling = (a, b)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = (x_train - b) / a
x_test = (x_test - b) / a
#%%
import matplotlib.pyplot as plt
for i in range(10):
    plt.subplot(1,2, 1)
    plt.imshow(x_test[i])
    plt.subplot(1,2,2)
    plt.imshow(y_test[i].reshape(3,3))
    plt.show()

# %%
