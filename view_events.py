
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Figureを追加
fig = plt.figure(figsize = (8, 8))

# 3DAxesを追加
ax = fig.add_subplot(111, projection='3d')

# Axesのタイトルを設定
ax.set_title("", size = 20)

# 軸ラベルを設定
ax.set_xlabel("x", size = 14, color = "r")
ax.set_ylabel("y", size = 14, color = "r")
ax.set_zlabel("z", size = 14, color = "r")

# 軸目盛を設定
ax.set_xticks([-5.0, -2.5, 0.0, 2.5, 5.0])
ax.set_yticks([-5.0, -2.5, 0.0, 2.5, 5.0])

# -5～5の乱数配列(100要素)
x = 10 * np.random.rand(100, 1) - 5
y = 10 * np.random.rand(100, 1) - 5
z = 10 * np.random.rand(100, 1) - 5

# 曲線を描画
ax.scatter(x, y, z, s = 40, c = "blue")

plt.show()