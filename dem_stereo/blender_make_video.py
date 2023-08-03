import bpy
import sys
import os
import numpy as np
import math
import mathutils
import shutil
import random

print("\n\n!!!!!program start!!!!!!!!\n")
random.seed(123)
dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)
from module.const_blender import *


def put_sun(theta_x=80, theta_y=0, theta_z=0):
    """
    太陽の設定を行う。
    """
    light = bpy.data.objects["Light"]
    light.data.type = "SUN"
    light.location = (0, 0, 0)
    light.data.energy = 10
    theta_x = math.radians(theta_x)  # 仰角に変換
    theta_y = math.radians(theta_y)  # 方位角に変換
    theta_z = math.radians(theta_z)  # 方位角に変換
    light.rotation_euler = (theta_x, theta_y, theta_z)
    return


def init(theta, camera_name_lst, video_dir_lst):
    """
    カメラ、太陽などの初期設定
    """

    # カメラの詳細設定
    camera_x = IMG_WIDTH
    camera_y = IMG_HEIGHT
    bpy.context.scene.render.resolution_x = camera_x
    bpy.context.scene.render.resolution_y = camera_y
    distance_between_camera = 1  # [m]
    viewing_radian = math.radians(38.2)  # カメラの視野角。
    radian_tilt = math.atan(distance_between_camera / CAM_Z) + viewing_radian / 2  # 左右のカメラをどのくらい傾けるか。ど真ん中に左右のカメラの端が来るようにする
    for camera_name in camera_name_lst:
        camera = bpy.data.objects[camera_name]
        camera.data.lens = FOCAL * 1000  # not [m] but [mm]
        camera.data.sensor_fit = "HORIZONTAL"
        camera.data.sensor_width = camera_x / 10
        camera.data.sensor_height = camera_y / 10
        camera.data.clip_end = 10
        # camera.location = (CAM_X, CAM_Y + (i - 1) * distance_between_camera, CAM_Z)

        camera.data.name = camera_name

    print(9999999999)
    for camera_name in camera_name_lst:
        camera = bpy.data.objects[camera_name]
        print(camera_name, camera.location)

    # カメラの標高座標の始点ト終点を定義
    x_start = CAM_X
    x_finish = x_start
    y_start = CAM_Y
    y_finish = y_start
    z_start = CAM_Z
    z_finish = 5.5
    z_length = z_start - z_finish
    velocity = 0.5  # 速度m/s
    video_fps = 100  # int(velocity*frame_num/z_length)
    # アニメーションのフレーム設定
    fram_start = 0
    frame_finish = int(z_length * video_fps / velocity)
    frame_num = frame_finish - fram_start
    frame_lst = [fram_start, frame_finish]
    z_lst = [z_start, z_finish]
    coodinate_lst = [(x_start, y_start, z_start), (x_finish, y_finish, z_finish)]
    # z_lst = [z_start, z_finish]
    bpy.context.scene.render.fps = video_fps
    bpy.context.scene.frame_end = fram_start
    bpy.context.scene.frame_end = frame_finish
    for camera_name in camera_name_lst:
        # フレームを挿入する
        bpy.context.scene.camera = bpy.data.objects[camera_name]
        for frame, z_frame in zip(frame_lst, z_lst):
            camera = bpy.data.objects[camera_name]
            print(camera_name, camera.location)
            bpy.context.scene.frame_set(frame)
            camera.rotation_mode = "ZXY"
            if camera_name == "Camera_left":
                camera.rotation_euler = (radian_tilt, 0, math.radians(90))
                camera.location = (CAM_X, CAM_Y + -1 * distance_between_camera, z_frame)
            elif camera_name == "Camera_center":
                camera.rotation_euler = (0, 0, math.radians(90))
                camera.location = (CAM_X, CAM_Y + 0 * distance_between_camera, z_frame)
            elif camera_name == "Camera_right":
                camera.rotation_euler = (-radian_tilt, 0, math.radians(90))
                camera.location = (CAM_X, CAM_Y + 1 * distance_between_camera, z_frame)

            # camera = bpy.data.objects[camera_name]
            print(camera_name, camera.location)
            camera.keyframe_insert(data_path="location", index=0)
            camera.keyframe_insert(data_path="location", index=1)
            camera.keyframe_insert(data_path="location", index=2)
            # print(camera_name)
            # print(camera.location)
            # カメラの移動を設定する
            camera = bpy.data.objects[camera_name]
            camera.location[2] = z_finish

        # for frame, (x, y, z) in zip(frame_lst, coodinate_lst):
        #     bpy.context.scene.frame_set(frame)
        #     camera.location = (x, y, z)
        #     camera.keyframe_insert(data_path="location", index=0)  # x
        #     camera.keyframe_insert(data_path="location", index=1)  # y
        #     camera.keyframe_insert(data_path="location", index=2)  # z

        camera = bpy.data.objects[camera_name]
        kf = camera.animation_data.action.fcurves[0].keyframe_points[0]  # アニメーション補間を線形に
        kf.interpolation = "LINEAR"

    ## アニメーション保存先設定
    for video_dir in video_dir_lst:
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        print(f"{video_dir=}")
        os.mkdir(video_dir)
    # bpy.context.scene.render.filepath = video_dir
    bpy.context.scene.render.image_settings.file_format = "AVI_JPEG"
    bpy.context.scene.render.image_settings.color_mode = "BW"

    # 太陽光の設定
    put_sun(theta_x=theta)


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
                        data[x, y] * METER_PER_GRID,
                    ]
                )
            )
    fIndexes = []
    for x in range(0, pix - 1):
        for y in range(0, pix - 1):
            fIndexes.append([x + y * pix, x + 1 + y * pix, x + 1 + (y + 1) * pix, x + (y + 1) * pix])

    mesh = bpy.data.meshes.new(object_name)
    mesh.from_pydata(verts, [], fIndexes)  # 点と面の情報からメッシュを生成

    obj = bpy.data.objects.new(object_name, mesh)  # メッシュ情報を新規オブジェクトに渡す
    bpy.context.scene.collection.objects.link(obj)  # オブジェクトをシーン上にリンク(v2.8)


def render(camera_name, save_video_path):
    """
    アニメーションし、動画を保存
    """
    bpy.context.scene.camera = bpy.data.objects[camera_name]
    bpy.data.scenes["Scene"].render.filepath = save_video_path
    bpy.ops.render.render(animation=True)


def remove(name):
    """
    物体を削除
    """
    targetob = bpy.data.objects.get(name)
    bpy.data.objects.remove(targetob)


if __name__ == "__main__":
    dem_path_abs = bpy.path.abspath(DEM_NP_PATH_BLENDER)  # https://twitter.com/Bookyakuno/status/1457726187745153038
    # video_path_abs = bpy.path.abspath(VIDEO_PATH_BLENDER)
    video_dir_lst = [
        bpy.path.abspath(VIDEO_LEFT_PATH_BLENDER),
        # bpy.path.abspath(VIDEO_CENTER_PATH_BLENDER),
        bpy.path.abspath(VIDEO_RIGHT_PATH_BLENDER),
    ]
    camera_name_lst = [
        "Camera_left",
        # "Camera_center",
        "Camera_right",
    ]
    THETA_X_SUN = 80  # 太陽の高度
    init(theta=THETA_X_SUN, camera_name_lst=camera_name_lst, video_dir_lst=video_dir_lst)
    DATA_NUM = 3000
    object_name = "dem"
    for i in range(DATA_NUM):
        number = str(i).zfill(5)
        dem_path = os.path.join(dem_path_abs, f"{number}.npy")
        a = 1
        theta_z = random.uniform(0, 360)
        put_sun(theta_x=THETA_X_SUN, theta_y=0, theta_z=theta_z)

        img2plot(dem_path)
        for camera_name, video_dir in zip(camera_name_lst, video_dir_lst):
            save_video_path = os.path.join(video_dir, f"{number}.avi")
            render(camera_name, save_video_path)

        remove(object_name)
