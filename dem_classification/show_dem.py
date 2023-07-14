import bpy
import numpy as np
import math
import mathutils
import time
import math
import os
import shutil
import pathlib
import sys

# aaa


def img2plot(np_path):
    """
    DEMモデルを作成する
    numpy配列をプロットし、面を作成。
    """
    data = np.load(np_path)
    pix, pix = data.shape
    verts = []
    for x in range(pix):
        for y in range(pix):
            verts.append(
                mathutils.Vector(
                    [
                        x * METER_PER_GRID,
                        y * METER_PER_GRID,
                        (data[x, y] * METER_PER_GRID),
                    ]
                )
            )
    fIndexes = []
    for x in range(0, pix - 1):
        for y in range(0, pix - 1):
            fIndexes.append(
                [x + y * pix, x + 1 + y * pix, x + 1 + (y + 1) * pix, x + (y + 1) * pix]
            )

    mesh = bpy.data.meshes.new(object_name)
    mesh.from_pydata(verts, [], fIndexes)  # 点と面の情報からメッシュを生成

    obj = bpy.data.objects.new(object_name, mesh)  # メッシュ情報を新規オブジェクトに渡す
    bpy.context.scene.collection.objects.link(obj)  # オブジェクトをシーン上にリンク(v2.8)


if __name__ == "__main__":
    filepath = bpy.data.filepath
    NOW_DIR = os.path.dirname(filepath)
    # SAVE_DIR = os.path.join(NOW_DIR,'video')
    print(NOW_DIR)
    # sys.path.append(NOW_DIR)
    # from module.const import *

    METER_PER_GRID = 0.1

    object_name = "dem"
    file_num = 0
    path = f"dem.npy"
    path = os.path.join(NOW_DIR, path)
    img2plot(path)
