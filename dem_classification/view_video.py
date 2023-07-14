from module import view
import cv2
import numpy as np
import matplotlib.pyplot as plt

n = int(input())
video_name = f"blender/video/{n:05}.avi"
image = view.get_first_frame(video_name)


# 画像のサイズを取得
height, width, _ = image.shape

# 枠の色と太さを指定
color = (255, 0, 0)  # 緑色
thickness = 1

# グリッドの行数と列数
rows = 3
cols = 3

# 枠の幅と高さを計算
cell_width = width // cols
cell_height = height // rows

# 画像にグリッドを描画
for i in range(1, rows):
    cv2.line(image, (0, i * cell_height), (width, i * cell_height), color, thickness)

for j in range(1, cols):
    cv2.line(image, (j * cell_width, 0), (j * cell_width, height), color, thickness)

# 描画結果の表示
plt.imshow(image)
plt.show()
