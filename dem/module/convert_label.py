import torch
import numpy as np
import matplotlib.pyplot as plt


class Dem2Img:
    """
    dem座標系のラベルをカメラのスクリーン座標系のラベルに変換
    """

    def __init__(
        self,
        focal,
        img_height,
        img_width,
        sensor_heitght,
        sensor_width,
        cam_x,
        cam_y,
        cam_z,
        meter_per_grid,
    ):
        """
        Blenderのカメラ初期値設定

        Parameters
        ----------
        focal : int
            カメラの焦点距離。基本0.050 m
        img_height, img_width : int
            ラベル付きスクリーン画像の縦横の長さ
        sensor_height, sensor_width : int
            センササイズの縦横長さ
        cam_x, cam_y, ca_z : any
            カメラの初期座標
        meter_per_grid : any
            blender状で１グリッド何mであるか
        """
        self.focal = focal
        self.img_height, self.img_width = img_height, img_width
        self.sensor_height, self.sensor_width = sensor_heitght, sensor_width
        self.cam_x, self.cam_y, self.cam_z = cam_x, cam_y, cam_z
        self.meter_per_grid = meter_per_grid

    def __call__(self, dem_label):
        """
        dem座標系のラベルをカメラのスクリーン座標系のラベルに変換

        Parameters
        ----------
        dem_label : torch.tensor
            デム座標系のラベル
        
        Return
        ---------
        label : torch.tensor
            スクリーン座標系のラベル
        """
        # https://qiita.com/S-Kaito/items/ace10e742227fd63bd4c
        self.dem_height, self.dem_width = dem_label.shape
        label = np.zeros((self.img_height, self.img_width))
        for x_pix in range(self.img_height):
            for y_pix in range(self.img_width):
                x_cam = (
                    (x_pix - self.img_height / 2) * self.sensor_height / self.img_height
                )  # スクリーン上のグローバル座標を求める。なお、原点は画像中心
                y_cam = (
                    (y_pix - self.img_width / 2) * self.sensor_width / self.img_width
                )

                dem_x_from_center = (
                    x_cam * self.cam_z / self.focal
                )  # デムのグローバル座標を求める。なお、原点は画像中心とする
                dem_y_from_center = y_cam * self.cam_z / self.focal

                dem_pix_x_from_center = (
                    dem_x_from_center // self.meter_per_grid
                )  # ピクセルに変換
                dem_pix_y_from_center = dem_y_from_center // self.meter_per_grid
                # dem_pix_x = dem_pix_x_from_center + self.dem_height // 2
                # dem_pix_y = dem_pix_y_from_center + self.dem_width // 2
                dem_pix_x = dem_pix_x_from_center + self.cam_x // self.meter_per_grid
                dem_pix_y = dem_pix_y_from_center + self.cam_y // self.meter_per_grid
                # print(dem_pix_x, dem_pix_y)
                # print(dem_label.shape)
                label[x_pix, y_pix] = dem_label[int(dem_pix_x), int(dem_pix_y)]
        return label


if __name__ == "__main__":
    dem_label = np.random.randint(0, 2, (128, 128))
    dd = Dem2Img()
    lable = dd(dem_label)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dem_label)
    ax2.imshow(lable)
    plt.show()
