
# %%
# import bpy
import numpy as np
import math
# import mathutils
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
%matplotlib inline
def resize_bilinear(src, hd, wd):
    # 出力画像用の配列生成（要素は全て空）
    dst = np.empty((hd, wd))

    # 元画像のサイズを取得
    h, w = src.shape[0], src.shape[1]

    # 拡大率を計算
    ax = wd / float(w)
    ay = hd / float(h)

    # バイリニア補間法
    for yd in tqdm(range(0, hd)):
        for xd in range(0, wd):
            x, y = xd/ax, yd/ay
            ox, oy = int(x), int(y)

            # 存在しない座標の処理
            if ox > w - 2:
                ox = w - 2
            if oy > h - 2:
                oy = h - 2

            # 重みの計算
            dx = x - ox
            dy = y - oy

            # 出力画像の画素値を計算
            # dst[yd][xd] = src[oy][ox] 
            dst[yd][xd] = max(src[oy][ox],src[oy+1][ox], src[oy-1][ox], src[oy][ox+1], src[oy][ox-1], )  

    return dst

def img2plot(path, all_pixel, partial_pixel, reshape):
    data = np.fromfile(path, dtype='>i2')
    data = data.reshape(all_pixel, all_pixel)
    data = data[:partial_pixel, :partial_pixel]
    print(np.min(data))
    # reshape = 100
    data = resize_bilinear(data, reshape, reshape)
    return data

def interporate(verts, ):
    # global df
    ERROR = -9999
    df = pd.DataFrame(data=verts)
    # df = df.fillna(ERROR)
    for col in tqdm(df.columns):
        df.loc[df[col] <= ERROR, col] = np.nan
    df = df.interpolate(limit_direction='both', )
    return df.values

#%%
#path = "C:/Users/oosim/Desktop/object-detection-by-SNN/dem/blender/kaguya/1.img"
path = "//kaguya/0.img" # 相対パスはスラッシュ二つにする必要があるらしい
path = "blender/kaguya/1.img" # 相対パスはスラッシュ二つにする必要があるらしい
# path = bpy.path.abspath(path)
RATE = 0.001
pixel = 12288
partial_pixel = pixel
reshape = pixel
# reshape = 500
meter_pixel = 7.403
meter_pixel *= RATE
verts = img2plot(path, pixel, partial_pixel, reshape)

#%%
ERROR = -9999
verts = np.nan_to_num(verts, nan=ERROR)
#%%
verts = interporate(verts)

plt.imshow(verts, cmap='gray')

plt.show()

np.save('blender/kaguya/np_all.npy', verts)



# %%
