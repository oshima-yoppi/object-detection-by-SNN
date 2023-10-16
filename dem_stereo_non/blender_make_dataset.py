import bpy
import sys
import os
import numpy as np
import math
import mathutils
import shutil
import random
import bpycv
import cv2
import numpy as np

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


def delete_object():
    """
    オブジェクトを削除する。
    """
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            bpy.data.objects.remove(obj)
    return


def put_camera(camera_name_lst):
    # 削除対象のカメラ以外のオブジェクトを探す
    exist_camera = [obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
    for camera in exist_camera:
        bpy.data.objects.remove(camera)
    delete_object()

    # カメラを配置する
    for camera_name in camera_name_lst:
        bpy.ops.object.camera_add()
        new_camera = bpy.context.object
        new_camera.name = camera_name


def setting(theta, camera_name_lst, camera_name_semaseg_lst, video_dir_lst):
    """
    カメラ、太陽などの初期設定
    """
    put_camera(camera_name_lst + camera_name_semaseg_lst)

    # カメラの詳細設定
    camera_x = IMG_WIDTH
    camera_y = IMG_HEIGHT
    bpy.context.scene.render.resolution_x = camera_x
    bpy.context.scene.render.resolution_y = camera_y
    distance_between_camera = DISTANCE_BETWEEN_CAMERA  # 左右のカメラの間の距離[m]
    viewing_radian = math.radians(CAMERA_THETA)  # カメラの視野角。
    overlap_length = OVERLAP_LENGTH  # 二つのカメラが見る範囲をどれくらい被らせるか[m](0.3m で高度1mの誤差、0.4m で高度1.5mの誤差、0.5m で高度2mの誤差、が許容できる)
    radian_tilt = (
        math.atan((distance_between_camera - overlap_length) / 2 / CAM_Z)
        + viewing_radian / 2
    )  # 左右のカメラをどのくらい傾けるか。
    for camera_name in camera_name_lst + camera_name_semaseg_lst:
        camera = bpy.data.objects[camera_name]
        camera.data.lens = FOCAL * 1000  # not [m] but [mm]
        camera.data.sensor_fit = "AUTO"
        # camera.data.sensor_fit = "HORIZONTAL"
        camera.data.sensor_width = SENSOR_WIDTH * 1000
        camera.data.sensor_height = SENSOR_HEIGHT * 1000
        camera.data.clip_end = 10
        # camera.location = (CAM_X, CAM_Y + (i - 1) * distance_between_camera, CAM_Z)

        # camera.data.name = camera_name

    for camera_name in camera_name_lst:
        camera = bpy.data.objects[camera_name]

    # カメラの標高座標の始点ト終点を定義 #
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

    camera_info = {}
    camera_info["Camera_left"] = [
        (CAM_X, CAM_Y + -1 * distance_between_camera / 2, z_start),
        (CAM_X, CAM_Y + -1 * distance_between_camera / 2, z_finish),
        (radian_tilt, 0, math.radians(90)),
    ]
    camera_info["Camera_center"] = [
        (CAM_X, CAM_Y + 0 * distance_between_camera, z_start),
        (CAM_X, CAM_Y + 0 * distance_between_camera, z_finish),
        (0, 0, math.radians(90)),
    ]
    camera_info["Camera_right"] = [
        (CAM_X, CAM_Y + 1 * distance_between_camera / 2, z_start),
        (CAM_X, CAM_Y + 1 * distance_between_camera / 2, z_finish),
        (-radian_tilt, 0, math.radians(90)),
    ]

    for camera_name in camera_name_lst:
        # フレームを挿入する
        bpy.context.scene.camera = bpy.data.objects[camera_name]
        # for frame, z_frame in zip(frame_lst, z_lst):
        for i, frame in enumerate(frame_lst):
            camera = bpy.data.objects[camera_name]
            print(camera_name, camera.location)
            bpy.context.scene.frame_set(frame)
            camera.rotation_mode = "ZXY"
            camera.rotation_euler = camera_info[camera_name][2]
            camera.location = camera_info[camera_name][i]

            # camera = bpy.data.objects[camera_name]
            # print(camera_name, camera.location)
            camera.keyframe_insert(data_path="location", index=0)
            camera.keyframe_insert(data_path="location", index=1)
            camera.keyframe_insert(data_path="location", index=2)
            # print(camera_name)
            # print(camera.location)
            # カメラの移動を設定する
            camera = bpy.data.objects[camera_name]
            camera.location[2] = z_finish

        camera = bpy.data.objects[camera_name]
        kf = camera.animation_data.action.fcurves[0].keyframe_points[0]  # アニメーション補間を線形に
        kf.interpolation = "LINEAR"
    # セマセグ用のカメラの位置設定
    for camera_name, camera_name_semaseg in zip(
        camera_name_lst, camera_name_semaseg_lst
    ):
        camera_semaseg = bpy.data.objects[camera_name_semaseg]
        camera_semaseg.rotation_mode = "ZXY"
        camera_semaseg.rotation_euler = camera_info[camera_name][2]
        camera_semaseg.location = camera_info[camera_name][0]

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


def img2plot(dem):
    """
    DEMモデルを作成する
    numpy配列をプロットし、面を作成。
    """

    pix, pix = dem.shape
    verts = []
    for x in range(pix):
        for y in range(pix):
            verts.append(
                mathutils.Vector(
                    [
                        x * METER_PER_GRID,
                        y * METER_PER_GRID,
                        dem[x, y] * METER_PER_GRID,
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


def render(camera_name, save_video_path):
    """
    アニメーションし、動画を保存
    """
    bpy.context.scene.camera = bpy.data.objects[camera_name]  #
    bpy.data.scenes["Scene"].render.filepath = save_video_path
    bpy.ops.render.render(animation=True)


def put_hazard(dem, raw_label):
    pix, pix = dem.shape
    bool_label = [[False] * (pix) for _ in range(pix)]  # 被り防止用の配

    verts = []
    faces = []
    num_of_square = 0
    # 1のラベルの領域を探索, 端っこは除く（めんどいから）
    for x in range(1, pix - 1):
        for y in range(1, pix - 1):
            if raw_label[x, y] == 1:
                # 4つの正方形を作成。正方形は左下の頂点を指定する。
                for v, w in ((0, 0), (-1, 0), (-1, -1), (0, -1)):
                    a, b = x + v, y + w  # 正方形の左下の頂点
                    if a < 0 or a >= pix or b < 0 or b >= pix:
                        continue
                    if bool_label[a][b]:
                        continue
                    bool_label[a][b] = True
                    for dx, dy in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        verts.append(
                            (
                                (a + dx) * METER_PER_GRID,
                                (b + dy) * METER_PER_GRID,
                                dem[a + dx, b + dy] * METER_PER_GRID,
                            )
                        )

                    faces.append(
                        (
                            4 * num_of_square,
                            4 * num_of_square + 1,
                            4 * num_of_square + 2,
                            4 * num_of_square + 3,
                        )
                    )
                    num_of_square += 1
    mesh = bpy.data.meshes.new("Plane_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new("Plane", mesh)
    obj.location = bpy.context.scene.cursor.location
    obj["inst_id"] = 1
    bpy.context.scene.collection.objects.link(obj)
    # del bool_label
    return


def render_semaseg(dem, raw_label, number, camera_name_semaseg_lst, label_dir_lst):
    put_hazard(dem, raw_label)
    for camera_name_semaseg, label_dir in zip(camera_name_semaseg_lst, label_dir_lst):
        camera = bpy.data.objects[camera_name_semaseg]
        bpy.context.scene.camera = camera
        result = bpycv.render_data()
        label = result["inst"]
        # print(np.max(label), np.min(label))
        save_label_path = os.path.join(label_dir, f"{number}.npy")
        np.save(save_label_path, label)
        # cv2.imwrite(os.path.join(label_dir, f"{str(num).zfill(5)}.png"), label)

    # delete_object()


def remove(name):
    """
    物体を削除
    """
    targetob = bpy.data.objects.get(name)
    bpy.data.objects.remove(targetob)


if __name__ == "__main__":
    dem_path_abs = bpy.path.abspath(
        DEM_NP_PATH_BLENDER
    )  # https://twitter.com/Bookyakuno/status/1457726187745153038
    # video_path_abs = bpy.path.abspath(VIDEO_PATH_BLENDER)
    label_dir_lst = [
        bpy.path.abspath(LABEL_LEFT_PATH_BLENDER),
        bpy.path.abspath(LABEL_CENTER_PATH_BLENDER),
        bpy.path.abspath(LABEL_RIGHT_PATH_BLENDER),
    ]
    video_dir_lst = [
        bpy.path.abspath(VIDEO_LEFT_PATH_BLENDER),
        bpy.path.abspath(VIDEO_CENTER_PATH_BLENDER),
        bpy.path.abspath(VIDEO_RIGHT_PATH_BLENDER),
    ]
    for dir in label_dir_lst + video_dir_lst:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
    camera_name_left = "Camera_left"
    camera_name_center = "Camera_center"
    camera_name_right = "Camera_right"
    camera_name_semaseg_left = "Camera_semaseg_left"
    camera_name_semaseg_center = "Camera_semaseg_center"
    camera_name_semaseg_right = "Camera_semaseg_right"
    camera_name_lst = [
        camera_name_left,
        camera_name_center,
        camera_name_right,
    ]  #
    camera_name_semaseg_lst = [
        camera_name_semaseg_left,
        camera_name_semaseg_center,
        camera_name_semaseg_right,
    ]
    THETA_X_SUN = 80  # 太陽の高度
    setting(
        theta=THETA_X_SUN,
        camera_name_lst=camera_name_lst,
        camera_name_semaseg_lst=camera_name_semaseg_lst,
        video_dir_lst=video_dir_lst,
    )
    DATA_NUM = 3000
    object_name = "dem"
    for i in range(DATA_NUM):
        # dem データの読み込み
        number = str(i).zfill(5)
        dem_path = os.path.join(dem_path_abs, f"{number}.npz")
        dem = np.load(dem_path)["dem"]
        raw_label = np.load(dem_path)["raw_label"]

        # a = 1
        theta_z = random.uniform(0, 360)
        put_sun(theta_x=THETA_X_SUN, theta_y=0, theta_z=theta_z)
        render_semaseg(
            dem=dem,
            number=number,
            raw_label=raw_label,
            camera_name_semaseg_lst=camera_name_semaseg_lst,
            label_dir_lst=label_dir_lst,
        )
        img2plot(dem)
        for camera_name, video_dir in zip(camera_name_lst, video_dir_lst):
            save_video_path = os.path.join(video_dir, f"{number}.avi")
            render(camera_name, save_video_path)

        delete_object()
