import bpy
import numpy as np
import math
import mathutils
import time

def resize_bilinear(src, hd, wd):
    # 出力画像用の配列生成（要素は全て空）
    dst = np.empty((hd, wd))

    # 元画像のサイズを取得
    h, w = src.shape[0], src.shape[1]

    # 拡大率を計算
    ax = wd / float(w)
    ay = hd / float(h)

    # バイリニア補間法
    for yd in range(0, hd):
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
            dst[yd][xd] = (1 - dx) * (1-dy) * src[oy][ox] + dx * (1-dy) * \
                src[oy][ox+1] + (1-dx) * dy * src[oy][ox+1] + \
                dx * dy * src[oy+1][ox+1]

    return dst

def img2plot(data):
    # data = np.load(path)
    pix, pix = data.shape
    verts = []
    for x in range(pix):
        for y in range(pix):
            verts.append(mathutils.Vector([x, y, (data[x, y])]))
            # verts.append(mathutils.Vector([np.uint8(x), np.uint8(y), np.uint8(data[x, y])]))
    return verts


start = time.time()
path = "//kaguya/mat_make1.npy" # 相対パスはスラッシュ二つにする必要があるらしい
path = bpy.path.abspath(path)

data = np.load(path)
pixel,_ = data.shape

verts = img2plot(path)
fIndexes = []  # 面のリスト

for x in range(0, pixel - 1):
    for y in range(0, pixel - 1):
        fIndexes.append([x + y * pixel,
                         x + 1 + y * pixel,
                         x + 1 + (y + 1) * pixel,
                         x + (y + 1) * pixel])

mesh = bpy.data.meshes.new('wave')
mesh.from_pydata(verts, [], fIndexes)  # 点と面の情報からメッシュを生成

obj = bpy.data.objects.new('wave', mesh)  # メッシュ情報を新規オブジェクトに渡す
bpy.context.scene.collection.objects.link(obj)  # オブジェクトをシーン上にリンク(v2.8)
print(time.time() - start)
# bpy.context.scene.objects.link(obj) #v2.7
obj.select = True  # 作ったオブジェクトを選択状態に