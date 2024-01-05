import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from module.const_blender import *
import os
import cv2
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, help="numver of dem")
args = parser.parse_args()
n = args.n

# DEM_DIR = "blender/dem"
# dem_path = os.path.join(DEM_DIR, f"{str(n).zfill(5)}.npz")

# # DEMの読み込み
# dem = np.load(dem_path)["dem"]
# print(dem.shape)
# # # サンプルデータを生成（適切なデータを置き換えてください）
# # w, H = 10, 10
# h, w = dem.shape


# # Meshgridを作成
# grid_meter = METER_PER_GRID
# dem = dem * grid_meter
# x = np.arange(0, w, 1)
# y = np.arange(0, h, 1)
# x, y = np.meshgrid(x, y)

# # 3Dプロットを作成
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# # プロット
# ax.plot_surface(x, y, dem, cmap="viridis")

# # ラベルの設定
# ax.set_xlabel("X軸")
# ax.set_ylabel("Y軸")
# ax.set_zlabel("標高")

# # グラフの表示
# plt.show()

vido_path = os.path.join(VIDEO_CENTER_PATH, f"{str(n).zfill(5)}.avi")
cap = cv2.VideoCapture(vido_path)
ret, frame = cap.read()
print(ret)
print(frame.shape)
fig = plt.figure()
plt.imshow(frame)
save_path = os.path.join("result_thesis", "den_normal_noize.pdf")
# plt.show()
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.savefig(save_path)

# pp = PdfPages(save_path)
# pp.savefig(fig)
