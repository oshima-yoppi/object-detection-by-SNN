import bpy
import numpy as np
import math
import mathutils
import time



def img2plot(np_path):
    data = np.load(path)
    pix, pix = data.shape
    verts = []
    for x in range(pix):
        for y in range(pix):
            verts.append(mathutils.Vector([x, y, (data[x, y])]))
            # verts.append(mathutils.Vector([np.uint8(x), np.uint8(y), np.uint8(data[x, y])]))
    return verts


start = time.time()

path = "//kaguya/mat_make.npy" # 相対パスはスラッシュ二つにする必要があるらしい
path = bpy.path.abspath(path)

pixel = 128


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