import bpy
import numpy as np
import math
import mathutils
import time


def img2plot(path, all_pixel, meter_pixel, reshape_pixel):
    data = np.load(path)
    print(data.min())
    verts = []
    for x in range(reshape_pixel):
        for y in range(reshape_pixel):
            verts.append(
                mathutils.Vector([x * meter_pixel, y * meter_pixel, RATE * data[x, y]])
            )
            # verts.append(mathutils.Vector([np.uint8(x), np.uint8(y), np.uint8(data[x, y])]))
    return verts


start = time.time()
# path = "C:/Users/oosim/Desktop/object-detection-by-SNN/dem/blender/kaguya/1.img"
path = "//kaguya/1.img"  # 相対パスはスラッシュ二つにする必要があるらしい
path = "//kaguya/np_all.npy"  # 相対パスはスラッシュ二つにする必要があるらしい
path = bpy.path.abspath(path)
RATE = 0.001
all_pixel = 12288
partial_pixel = all_pixel
reshape_pixel = 1000
meter_pixel = 7.403
meter_pixel *= RATE
meter_pixel *= partial_pixel / reshape_pixel
verts = img2plot(path, all_pixel, meter_pixel, reshape_pixel)
fIndexes = []  # 面のリスト

for x in range(0, reshape_pixel - 1):
    for y in range(0, reshape_pixel - 1):
        fIndexes.append(
            [
                x + y * reshape_pixel,
                x + 1 + y * reshape_pixel,
                x + 1 + (y + 1) * reshape_pixel,
                x + (y + 1) * reshape_pixel,
            ]
        )

mesh = bpy.data.meshes.new("wave")
mesh.from_pydata(verts, [], fIndexes)  # 点と面の情報からメッシュを生成

obj = bpy.data.objects.new("wave", mesh)  # メッシュ情報を新規オブジェクトに渡す# error heare
bpy.context.scene.collection.objects.link(obj)  # オブジェクトをシーン上にリンク(v2.8)
print(time.time() - start)
# bpy.context.scene.objects.link(obj) #v2.7
obj.select = True  # 作ったオブジェクトを選択状態に
