import tensorflow as tf
from tensorflow.keras.layers import Layer


class CustomPaddingLayer(Layer):
    def __init__(self, padding=(0, 0), **kwargs):
        super(CustomPaddingLayer, self).__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        padded_inputs = tf.pad(
            inputs,
            paddings=[[0, 0], [self.padding[0], self.padding[0]], [self.padding[1], self.padding[1]], [0, 0]],
            mode="CONSTANT",
            constant_values=0,
        )
        return padded_inputs


# ダミーデータの生成
input_data = tf.random.normal((1, 4, 4, 3))  # (バッチサイズ, 高さ, 幅, チャンネル数)

# カスタムパディングレイヤーの作成
custom_padding_layer = CustomPaddingLayer(padding=(3, 2))

# パディングの適用
padded_data = custom_padding_layer(input_data)
print(input_data.shape)
print(padded_data.shape)
