# %%
from tensorflow import keras
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# from google.colab.patches import cv2_imshow #cv2_imshow(img)


# %%
# Load MNIST dataset
INPUT_HEIGHT, INPUT_WIDTH = 50, 50
INPUT_HEIGHT, INPUT_WIDTH = 200, 200
INPUT_HEIGHT, INPUT_WIDTH = 130, 162
COUNT = False
# COUNT = True
RESIZE = True
# dataset_path = f"dataset/dataset_{INPUT_WIDTH}_{INPUT_HEIGHT}_count-{COUNT}.pickle"
dataset_path = f"dataset.pickle"

with open(dataset_path, "rb") as f:
    train_lst, label_lst = pickle.load(f)
##
train_lst = np.array(train_lst)
label_lst = np.array(label_lst)
print(train_lst.shape)
num_train = len(train_lst)
x_train, x_test, y_train, y_test = train_test_split(train_lst, label_lst, shuffle=False, test_size=0.2, random_state=42)
x_train.shape
x_train.max()

# %%
# Reshape x-data
x_train = x_train.reshape(-1, INPUT_HEIGHT, INPUT_WIDTH, 1)
x_test = x_test.reshape(-1, INPUT_HEIGHT, INPUT_WIDTH, 1)

# Set aside raw test data for use with Akida runtime later
raw_x_test = x_test.astype("uint8")
raw_y_test = y_test.astype("uint8")

# Rescale x-data
if COUNT:
    a = 100  # alltime=1000 ms frame v2e = 10 ms
    # a = 30 # alltime=1000 ms frame v2e = 10 ms
    a = 1
    b = 0
else:
    a = 1
    b = 0
input_scaling = (a, b)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = (x_train - b) / a
x_test = (x_test - b) / a
# %%
x_train.max()
# %%
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(x_train[20])
ax2.imshow(y_train[20].reshape(3, 3))
# print(x_train[0].shape,y_train[0].shape)
ax3.hist(x_train[0].reshape(-1))
# plt.show()
# %%

drop_rate = 0.2
# model_keras = keras.models.Sequential(
#     [
#         keras.layers.Conv2D(
#             filters=16,
#             kernel_size=3,
#             input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1),
#             strides=1,
#         ),
#         keras.layers.BatchNormalization(),
#         keras.layers.ReLU(),
#         keras.layers.Dropout(drop_rate),
#         keras.layers.Conv2D(filters=32, kernel_size=5, strides=2),
#         keras.layers.BatchNormalization(),
#         keras.layers.ReLU(),
#         keras.layers.Dropout(drop_rate),
#         keras.layers.Conv2D(filters=64, kernel_size=5, strides=2),
#         keras.layers.BatchNormalization(),
#         keras.layers.ReLU(),
#         keras.layers.Dropout(drop_rate),
#         keras.layers.Conv2D(filters=32, kernel_size=5, strides=2),
#         keras.layers.BatchNormalization(),
#         keras.layers.ReLU(),
#         keras.layers.Conv2D(filters=1, kernel_size=1, strides=1),
#         keras.layers.BatchNormalization(),
#     ],
# )
model_keras = keras.models.Sequential(
    [
        keras.layers.Conv2D(
            filters=16,
            kernel_size=5,
            input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1),
            strides=2,
        ),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(drop_rate),
        keras.layers.Conv2D(filters=32, kernel_size=5, strides=2),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(drop_rate),
        keras.layers.Conv2D(filters=64, kernel_size=5, strides=2),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.Dropout(drop_rate),
        keras.layers.Conv2D(filters=1, kernel_size=5, strides=2),
        keras.layers.BatchNormalization(),
        # keras.layers.ReLU(),
        # keras.layers.Dropout(drop_rate),
        # keras.layers.Conv2D(filters=1, kernel_size=5, strides=2),
        # keras.layers.BatchNormalization(),
        # keras.layers.ReLU(),
        # keras.layers.Dropout(drop_rate),
        # keras.layers.Conv2D(filters=1, kernel_size=1, strides=1),
        # keras.layers.BatchNormalization(),
    ],
)

model_keras.summary()

# %%
from cnn2snn import check_model_compatibility

print("Model compatible for Akida conversion:", check_model_compatibility(model_keras))
# %%
# model_keras.compile(
#     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer='adam',
#     metrics=['accuracy'])
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.layers import Layer


class CustomPaddingLayer(Layer):
    def __init__(self, **kwargs):
        super(CustomPaddingLayer, self).__init__(**kwargs)

    def call(self, inputs, padding=(1, 1)):
        padded_inputs = tf.pad(
            inputs,
            paddings=[[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]],
            mode="CONSTANT",
            constant_values=0,
        )
        return padded_inputs


def change_output(x):
    """
    出力サイズを（３，３）にするための関数
    """
    x = K.sigmoid(x)
    # print(x.shape)
    _, h, w, _ = x.shape
    padding_h = h % 3
    padding_w = w % 3
    stride_h = (h + padding_h * 2) // 3
    stride_w = (w + padding_w * 2) // 3
    paddinger = CustomPaddingLayer()
    x = paddinger(x, padding=(padding_h, padding_w))
    print(x.shape)
    print(f"padding_h:{padding_h},padding_w:{padding_w},stride_h:{stride_h},stride_w:{stride_w}")
    x = tf.keras.layers.MaxPool2D(pool_size=(stride_h, stride_w), strides=(stride_h, stride_w))(x)
    print(x.shape)
    # x = tf.image.resize(x, [3, 3])
    return x


def Dice(targets, inputs, smooth=1e-6):
    # targets = targets.astye('float')
    # flatten label and prediction tensors
    # tf_show(inputs[0])
    batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)
    inputs = K.softmax(inputs, axis=-1)
    inputs = K.flatten(inputs[:, :, :, 1])
    targets = K.flatten(targets)
    intersection = tf.reduce_sum(inputs * targets)
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return dice


def DiceLoss(targets, inputs, smooth=1e-6):
    # targets = targets.astye('float')
    # flatten label and prediction tensors
    # tf_show(inputs[0])
    batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)

    inputs = change_output(inputs)
    # inputs = K.softmax(inputs, axis=-1)
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    intersection = tf.reduce_sum(inputs * targets)  # https://tensorflow.classcat.com/2018/09/07/tensorflow-tutorials-images-segmentation/
    # print(targets.shape, inputs.shape)
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


def weighted_focal_Loss(targets, inputs, beta=0.6, smooth=1e-6):
    # targets = targets.astye('float')
    # flatten label and prediction tensors
    # tf_show(inputs[0])
    batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)
    # inputs = K.softmax(inputs, axis=-1)
    inputs = change_output(inputs)
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    intersection = tf.reduce_sum(inputs * targets)
    precision = intersection / (K.sum(inputs) + smooth)
    recall = intersection / (K.sum(targets) + smooth)
    f = ((1 + beta**2) * precision * recall + smooth) / (beta**2 * precision + recall + smooth)
    return 1 - f


def IoU(targets, inputs, smooth=1e-6):
    batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)
    # inputs = K.softmax(inputs, axis=-1)
    inputs = change_output(inputs)
    inputs = K.flatten(inputs[:, :, :, 1])
    targets = K.flatten(targets)
    intersection = tf.reduce_sum(inputs * targets)
    iou = (intersection + smooth) / (K.sum(targets) + K.sum(inputs) - intersection + smooth)
    return iou


def IoU_Loss(targets, inputs, smooth=1e-6):
    batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)
    # inputs = K.softmax(inputs, axis=-1)
    inputs = change_output(inputs)
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    intersection = tf.reduce_sum(inputs * targets)
    iou = (intersection + smooth) / (K.sum(targets) + K.sum(inputs) - intersection + smooth)
    return 1 - iou


def IoU_eval(targets, inputs, smooth=1e-6):
    batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)
    inputs = change_output(inputs)
    inputs = K.flatten(inputs)
    inputs = tf.where(inputs >= 0.5, 1.0, 0.0)
    targets = K.flatten(targets)
    intersection = tf.reduce_sum(inputs * targets)
    iou = (intersection + smooth) / (K.sum(targets) + K.sum(inputs) - intersection + smooth)
    return iou


def recall_eval(targets, inputs, smooth=1e-6):
    # batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)
    inputs = change_output(inputs)
    inputs = K.flatten(inputs)
    inputs = tf.where(inputs >= 0.5, 1.0, 0.0)
    targets = K.flatten(targets)
    recall = (K.sum(inputs * targets) + smooth) / (K.sum(targets) + smooth)
    # intersection = tf.reduce_sum(inputs * targets)
    # iou = (intersection + smooth) / (K.sum(targets) + K.sum(inputs) - intersection + smooth)
    return recall


def precission_eval(targets, inputs, smooth=1e-6):
    # batch = len(inputs)
    targets = tf.cast(targets, dtype=tf.float32)
    inputs = change_output(inputs)
    inputs = K.flatten(inputs)
    inputs = tf.where(inputs >= 0.5, 1.0, 0.0)
    targets = K.flatten(targets)
    precission = (K.sum(inputs * targets) + smooth) / (K.sum(inputs) + smooth)
    # intersection = tf.reduce_sum(inputs * targets)
    # iou = (intersection + smooth) / (K.sum(targets) + K.sum(inputs) - intersection + smooth)
    return precission


# def IoU_(targets, inputs, smooth=1e-6):
#     batch = len(inputs)
#     targets = tf.cast(targets, dtype=tf.float32)
#     inputs = change_output(inputs)
#     inputs = K.flatten(inputs[:,:,:,1])
#     targets = K.flatten(targets)
#     iou = iou_metric(targets, inputs)
#     return  iou
optimizer = keras.optimizers.Adam(learning_rate=0.01)
model_keras.compile(loss=DiceLoss, optimizer=optimizer, metrics=[precission_eval, recall_eval])
# model_keras.compile(loss=weighted_focal_Loss, optimizer=optimizer, metrics=[precission_eval, recall_eval])
model_keras.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=20)


# score = model_keras.evaluate(x_test, y_test, verbose=0)
# print('Test accuracy:', score[1])
# %%
print(x_test.shape)
score = model_keras.evaluate(x_test, y_test, verbose=0)
print("Test accuracy:", score[1])
# %%
# for i in range(10):
#     out = model_keras.predict(x_test[i : i + 1])
#     pre = change_output(out)[0]
#     plt.subplot(1, 3, 1)
#     plt.imshow(x_test[i])
#     plt.subplot(1, 3, 2)
#     plt.imshow(pre, vmin=0, vmax=1)
#     plt.subplot(1, 3, 3)
#     plt.imshow(y_test[i].reshape(3, 3), vmin=0, vmax=1)
#     plt.show()
# out = model_keras.predict(x_test[1:2])
# # %%
# for i in range(10):
#     out = model_keras.predict(x_train[i : i + 1])
#     pre = change_output(out)[0]
#     plt.subplot(1, 3, 1)
#     plt.imshow(x_train[i])
#     plt.subplot(1, 3, 2)
#     plt.imshow(pre, vmin=0, vmax=1)
#     plt.subplot(1, 3, 3)
#     plt.imshow(y_train[i].reshape(3, 3), vmin=0, vmax=1)
#     plt.show()

# %%
