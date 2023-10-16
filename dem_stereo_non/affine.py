import cv2
import numpy as np
import matplotlib.pyplot as plt

# from module import view
from module.const_blender import *
from module.view import *
import os
import math


def change(distance_between_camera=0.1, viewing_angle=38.2, focal=0.05, y=0.0342 / 2, h=CAM_Z, which="right"):
    # https://daily-tech.hatenablog.com/entry/2016/05/29/023229
    if which == "right":
        focal = 0.05
        viewing_radian = math.radians(viewing_angle)  # カメラの視野角。
        radian_tilt = math.atan(distance_between_camera / CAM_Z) + viewing_radian / 2  # 左右のカメラをどのくらい傾けるか。ど真ん中に左右のカメラの端が来るようにする
        # y = 0.0342 / 2  # ??
        radian_tilt = math.pi / 2 - radian_tilt
        sin = math.sin(radian_tilt)
        cos = math.cos(radian_tilt)
        ans = (y * focal * h * sin + focal**2 * (-h * cos + distance_between_camera * sin)) / (
            focal * (h * sin + distance_between_camera * cos) + y * h * cos
        )
        print(ans)
    return ans


def change_from_TopView(y, h, focal, distance_between_camera, theta):
    """
    鳥瞰図からカメラの位置を変えたときに、画像の中心がどれだけずれるかを計算する.
    y軸のみを考える。
    https://daily-tech.hatenablog.com/entry/2016/05/29/023229を参考にした。
    このurl では、カメラの位置を変えたときに、画像の中心がどれだけずれるかを計算している。urlの逆のことを行っている。

    Parameters
    ----------
    y : float
        カメラのy座標。上記url のY`に相当する。
    h : float
        カメラの高さ
    focal : float
        カメラの焦点距離[m]
    distance_between_camera : float
        カメラ間の距離[m]
    theta : float
        カメラの傾き[rad]。上記url のθに相当する。

    Returns
    -------
    ans : float
        もう一方のカメラ画像のどこに移るか。画像の中心からの距離[m]。上記url のYに相当する。

    """
    sin = math.sin(theta)
    cos = math.cos(theta)
    ans = (y * focal * h * sin + focal**2 * (-h * cos + distance_between_camera * sin)) / (
        focal * (h * sin + distance_between_camera * cos) + y * h * cos
    )
    # print(ans)
    return ans


# 右側の計算
print(math.degrees(math.atan(0.0342 / 2 / 0.05)))
focal = 0.05
viewing_angle = 38.2
distance_between_camera = 2
viewing_radian = math.radians(viewing_angle)  # カメラの視野角。
radian_tilt = math.atan(distance_between_camera / CAM_Z) + viewing_radian / 2  # 左右のカメラをどのくらい傾けるか。ど真ん中に左右のカメラの端が来るようにする
print(88)
print(math.degrees(radian_tilt), "=", math.degrees(math.atan(distance_between_camera / CAM_Z)), "+", math.degrees(viewing_radian / 2))
# radian_tilt = math.radians(28.6)
radian_tilt = math.pi / 2 - radian_tilt

delta_right = change_from_TopView(y=0.0342 / 2, h=CAM_Z, focal=focal, distance_between_camera=distance_between_camera, theta=radian_tilt)
print(delta_right)


# 左側の計算
focal = 0.05
viewing_angle = 38.2
distance_between_camera = 2
viewing_radian = math.radians(viewing_angle)  # カメラの視野角。
radian_tilt = math.atan(distance_between_camera / CAM_Z) + viewing_radian / 2  # 左右のカメラをどのくらい傾けるか。ど真ん中に左右のカメラの端が来るようにする
radian_tilt *= -1
radian_tilt = math.pi / 2 - radian_tilt
print(math.degrees(radian_tilt))
delta_left = change_from_TopView(y=-0.0342 / 2, h=CAM_Z, focal=focal, distance_between_camera=-distance_between_camera, theta=radian_tilt)
print(delta_left)

i = 0
while True:
    number = i
    center_path = os.path.join(VIDEO_CENTER_PATH, f"{str(number).zfill(5)}.avi")
    left_path = os.path.join(VIDEO_LEFT_PATH, f"{str(number).zfill(5)}.avi")
    right_path = os.path.join(VIDEO_RIGHT_PATH, f"{str(number).zfill(5)}.avi")

    center_img = get_first_frame(center_path)
    left_img = get_first_frame(left_path)
    right_img = get_first_frame(right_path)

    width_right = right_img.shape[1]
    idx_right = width_right // 2 - delta_right * 1000
    idx_right = int(idx_right)

    width_left = left_img.shape[1]
    idx_left = width_left // 2 - delta_left * 1000
    idx_left = int(idx_left)

    print(idx_left)
    plt.subplot(1, 5, 1)
    plt.imshow(center_img)
    plt.subplot(1, 5, 2)
    plt.imshow(left_img)
    plt.subplot(1, 5, 3)
    plt.imshow(right_img)
    plt.subplot(1, 5, 4)
    plt.imshow(right_img[:, 182:])
    plt.subplot(1, 5, 5)
    plt.imshow(left_img[:, :160])
    plt.show()
    i += 1
