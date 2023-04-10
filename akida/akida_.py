#%%import cnn2snn
from tensorflow import keras
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
# from google.colab.patches import cv2_imshow #cv2_imshow(img)
from keras.layers.serialization import activation
import tensorflow as tf
import keras.backend as K
#%%
# from cnn2snn import check_model_compatibility
from cnn2snn import convert
from cnn2snn import quantize
#%%
dataset_path = "dataset/dataset.pickle"
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
#%%
model_akida = convert(model_quantized, input_scaling=input_scaling)