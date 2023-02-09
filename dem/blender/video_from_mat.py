import bpy
import numpy as np
import math
import mathutils
import time
# import math
import os
import shutil
import pathlib
# aaa

def init(theta, save_dir):
    """
    カメラ、太陽などの初期設定
    """

    # カメラの詳細設定
    camera_x = 346
    camera_y = 260
    bpy.context.scene.render.resolution_x = camera_x
    bpy.context.scene.render.resolution_y = camera_y
    camera = bpy.data.objects['Camera']
    camera.data.lens = 50
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.sensor_width = camera_x/10
    camera.data.sensor_height = camera_y/10
    camera.data.clip_end = 400
    camera.location = (64, 64, 300)
    camera.rotation_euler = (0, 0, math.radians(90))

    
    # カメラの標高座標の始点ト終点を定義
    z_start = 164
    z_finish = 154
    z_length = z_start - z_finish
    velocity = 10 # 速度m/s
    video_fps =  100#int(velocity*frame_num/z_length)
    # アニメーションのフレーム設定
    fram_start = 0
    frame_finish = int(z_length*video_fps/velocity)
    frame_num = frame_finish - fram_start
    frame_lst = [fram_start,frame_finish]
    z_lst = [z_start, z_finish]
    bpy.context.scene.render.fps = video_fps
    bpy.context.scene.frame_end = fram_start
    bpy.context.scene.frame_end = frame_finish
    #フレームを挿入する
    for frame, z in zip(frame_lst, z_lst):
        bpy.context.scene.frame_set(frame)
        camera.location[2] = z
        camera.keyframe_insert(data_path = "location",index = 2)
    kf = camera.animation_data.action.fcurves[0].keyframe_points[0]# アニメーション補間を線形に
    kf.interpolation = 'LINEAR' 

    ## アニメーション保存先設定
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    print(f'{save_dir=}')
    os.mkdir(save_dir)
    bpy.context.scene.render.filepath = save_dir
    bpy.context.scene.render.image_settings.file_format = 'AVI_JPEG'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    


    # 太陽光の設定
    light = bpy.data.objects['Light']
    light.data.type = 'SUN'
    light.location = (0,0,0)
    light.data.energy = 10
    theta = math.radians(90-theta) # 仰角に変換
    light.rotation_euler = (theta,0,0)


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
            verts.append(mathutils.Vector([x, y, (data[x, y])]))
    fIndexes = []
    for x in range(0, pix - 1):
        for y in range(0, pix - 1):
            fIndexes.append([x + y * pix,
                            x + 1 + y * pix,
                            x + 1 + (y + 1) * pix,
                            x + (y + 1) * pix])

    mesh = bpy.data.meshes.new(object_name)
    mesh.from_pydata(verts, [], fIndexes)  # 点と面の情報からメッシュを生成

    obj = bpy.data.objects.new(object_name, mesh)  # メッシュ情報を新規オブジェクトに渡す
    bpy.context.scene.collection.objects.link(obj)  # オブジェクトをシーン上にリンク(v2.8)
    
   

def render(save_path):
    """
    アニメーションし、動画を保存
    """
    bpy.context.scene.camera = bpy.data.objects["Camera"]
    bpy.data.scenes["Scene"].render.filepath = save_path
    bpy.ops.render.render(animation=True)

def remove(name):
    """
    物体を削除
    """
    targetob = bpy.data.objects.get(name)
    bpy.data.objects.remove(targetob)



if __name__ == "__main__":
    filepath = bpy.data.filepath
    NOW_DIR = os.path.dirname(filepath)
    SAVE_DIR = os.path.join(NOW_DIR,'video')
    init(theta = 10, save_dir=SAVE_DIR)



    DATA_NUM = 3001
    object_name = "dem"
    for i in range(DATA_NUM):
        path = f"dem/dem_{i}.npy" 
        path = os.path.join(NOW_DIR, path)
        img2plot(path)

        save_path = os.path.join(SAVE_DIR, f'{str(i).zfill(5)}.avi')
        render(save_path)


        remove(object_name)




