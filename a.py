import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 200x200のランダムなデータを生成する例
pix = 5
data = np.random.rand(pix, pix)
# 3Dグラフの初期化
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# XとY座標のメッシュグリッドを作成
x = np.arange(0, pix)
y = np.arange(0, pix)
x, y = np.meshgrid(x, y)

# 3Dメッシュプロットを作成
ax.plot_surface(x, y, data, cmap="viridis")

# ラベルを設定
ax.set_xlabel("X軸")
ax.set_ylabel("Y軸")
ax.set_zlabel("Z軸")

# グラフを表示
plt.show()
