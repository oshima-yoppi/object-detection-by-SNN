#%%
from tensorflow import keras
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
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

y_train = tf.squeeze(y_train)
y_test = tf.squeeze(y_test)
#%%
import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(x_train[2])
ax2.imshow(y_train[2].reshape(3,3))
# print(x_train[0].shape,y_train[0].shape)
ax3.hist(x_train[0].reshape(-1))
# plt.show()
# %%
model_keras = keras.models.Sequential([
    keras.layers.Conv2D(
        filters=16, kernel_size=5, input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1), strides=1, ),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Conv2D(
        filters=32, kernel_size=5, strides=2),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Conv2D(
        filters=64, kernel_size=5, strides=2),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Conv2D(
        filters=128, kernel_size=5, strides=2),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Conv2D(
        filters=32, kernel_size=1, strides=1),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Conv2D(
        filters=2, kernel_size=1, strides=1),
],)




model_keras.summary()
# %%
from cnn2snn import check_model_compatibility

print("Model compatible for Akida conversion:",
      check_model_compatibility(model_keras))
# %%
# model_keras.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer='adam',
#     metrics=['accuracy'])
import tensorflow as tf
import keras.backend as K

def change_output(x):
    # print(x.shape)
    # print(x.shape)
    x = K.softmax(x, axis=-1)
    x = x[:,:,:,1]
    x = tf.expand_dims(x, axis=-1)
    x = tf.image.resize(x, [3,3])
    # print(x.shape)
    # x = K.sigmoid(x-0.25)
    return x
def Dice(targets, inputs, smooth=1e-6):
    # targets = targets.astye('float')
    #flatten label and prediction tensors
    # tf_show(inputs[0])
    batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)
    inputs = K.softmax(inputs, axis=-1)
    inputs = K.flatten(inputs[:,:,:,1])
    targets = K.flatten(targets)
    intersection = tf.reduce_sum(inputs*targets)
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return  dice
def DiceLoss(targets, inputs, smooth=1e-6):
    # targets = targets.astye('float')
    #flatten label and prediction tensors
    # tf_show(inputs[0])
    batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)

    inputs = change_output(inputs)
    # inputs = K.softmax(inputs, axis=-1)
    # print(inputs.shape, targets.shape)
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    intersection = tf.reduce_sum(inputs*targets) # https://tensorflow.classcat.com/2018/09/07/tensorflow-tutorials-images-segmentation/
    # print(targets.shape, inputs.shape)
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice
iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)
def IoU(targets, inputs, smooth=1e-6):
    batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)
    # inputs = K.softmax(inputs, axis=-1)
    inputs = change_output(inputs)
    inputs = K.flatten(inputs[:,:,:,1])
    targets = K.flatten(targets)
    intersection = tf.reduce_sum(inputs*targets)
    iou = (intersection + smooth)/(K.sum(targets) + K.sum(inputs) - intersection + smooth)
    return  iou
def IoU_Loss(targets, inputs, smooth=1e-6):
    batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)
    # inputs = K.softmax(inputs, axis=-1)
    inputs = change_output(inputs)
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    intersection = tf.reduce_sum(inputs*targets)
    iou = (intersection + smooth)/(K.sum(targets) + K.sum(inputs) - intersection + smooth)
    return  1 - iou
def IoU_eval(targets, inputs, smooth=1e-6):
    batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)
    inputs = change_output(inputs)
    inputs = K.flatten(inputs)
    inputs = tf.where(inputs>=0.5, 1.0, 0.0)
    targets = K.flatten(targets)
    intersection = tf.reduce_sum(inputs*targets)
    iou = (intersection + smooth)/(K.sum(targets) + K.sum(inputs) - intersection + smooth)
    return  iou
# def IoU_(targets, inputs, smooth=1e-6):
#     batch = len(inputs)
#     targets = tf.cast(targets, dtype=tf.float32)
#     inputs = change_output(inputs)
#     inputs = K.flatten(inputs[:,:,:,1])
#     targets = K.flatten(targets)
#     iou = iou_metric(targets, inputs)
#     return  iou
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model_keras.compile(
    loss= DiceLoss,
    optimizer=optimizer,
    metrics=[IoU_eval]
    )
model_keras.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=30)


# score = model_keras.evaluate(x_test, y_test, verbose=0)
# print('Test accuracy:', score[1])
# %%
print(x_test.shape)
score = model_keras.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])
# %%

for i in range(10):
    out = model_keras.predict(x_test[i:i+1])
    print(out.shape)
    pre = change_output(out[0])
    print(pre.shape)
    plt.subplot(1,3,1)
    plt.imshow(x_test[i])
    plt.subplot(1,3,2)  
    plt.imshow(pre, vmin=0, vmax=1)
    plt.subplot(1,3,3)
    plt.imshow(y_test[i].reshape(3,3), vmin=0, vmax=1)
    plt.show()
out = model_keras.predict(x_test[1:2])
# %%
for i in range(10):
    out = model_keras.predict(x_train[i:i+1])
    pre = change_output(out[0])
    plt.subplot(1,3,1)
    plt.imshow(x_train[i])
    plt.subplot(1,3,2)
    plt.imshow(pre, vmin=0, vmax=1)
    plt.subplot(1,3,3)
    plt.imshow(y_train[i].reshape(3,3), vmin=0, vmax=1)
    plt.show()

# %%