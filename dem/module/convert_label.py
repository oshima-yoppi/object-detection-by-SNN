import torch
import numpy as np
import matplotlib.pyplot as plt
class Dem2Img():
    """
    dem座標系のラベルをカメラのスクリーン座標系のラベルに変換
    """
    def __init__(self,focal, img_height, img_width, sensor_heitght, sensor_width, cam_x, cam_y, cam_z):
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
        """
        self.focal = focal
        self.img_height, self.img_width = img_height, img_width
        self.sensor_height, self.sensor_width = sensor_heitght, sensor_width
        self.cam_x, self.cam_y, self.cam_z = cam_x, cam_y, cam_z
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
        initial_value = 100
        label = np.full((self.img_height, self.img_width), initial_value) 
        for x_pix in range(self.img_height):
            for y_pix in range(self.img_width):
                x_cam = (x_pix - self.img_height/2)*self.sensor_height/self.img_height# 座標を求める
                y_cam = (y_pix - self.img_width/2)*self.sensor_width/self.img_width

                dem_x = x_cam*self.cam_z/self.focal
                dem_y = y_cam*self.cam_z/self.focal

                dem_pix_x = int(dem_x + self.dem_height/2)
                dem_pix_y = int(dem_y + self.dem_width/2)
                
                label[x_pix, y_pix] = dem_label[dem_pix_x, dem_pix_y]
        return label

# ゴミプログラム
# class Dem2Img():
#     """
#     dem座標系のラベルをカメラのスクリーン座標系のラベルに変換
#     """
#     def __init__(self,focal, img_height, img_width, sensor_heitght, sensor_width, cam_x, cam_y, cam_z):
#         """
#         Blenderのカメラ初期値設定

#         Parameters
#         ----------
#         focal : int
#             カメラの焦点距離。基本0.050 m
#         img_height, img_width : int
#             ラベル付きスクリーン画像の縦横の長さ
#         sensor_height, sensor_width : int
#             センササイズの縦横長さ
#         cam_x, cam_y, ca_z : any
#             カメラの初期座標
#         """
#         self.focal = focal
#         self.img_height, self.img_width = img_height, img_width
#         self.sensor_height, self.sensor_width = sensor_heitght, sensor_width
#         self.cam_x, self.cam_y, self.cam_z = cam_x, cam_y, cam_z
#     def __call__(self, dem_label):
#         """
#         dem座標系のラベルをカメラのスクリーン座標系のラベルに変換

#         Parameters
#         ----------
#         dem_label : torch.tensor
#             デム座標系のラベル
        
#         Return
#         ---------
#         label : torch.tensor
#             スクリーン座標系のラベル
#         """
#         # https://qiita.com/S-Kaito/items/ace10e742227fd63bd4c
#         self.dem_height, self.dem_width = dem_label.shape
#         label = np.zeros((self.img_height, self.img_width)) 
#         for x_world in range(self.dem_height):
#             for y_world in range(self.dem_width):
#                 if dem_label[x_world, y_world]:
#                     x_cam = x_world - self.cam_x
#                     y_cam = y_world - self.cam_y
#                     x_img = self.focal*x_cam/self.cam_z # 個々のZを高度に変更すること
#                     y_img = self.focal*y_cam/self.cam_z
                    
#                     x_img += self.sensor_height/2
#                     y_img += self.sensor_width / 2
#                     x_pix = int(x_img /self.sensor_height* self.img_height)
#                     y_pix = int(y_img /self.sensor_width * self.img_width)
             
#                     if 0 <= x_pix < self.img_height and 0 <= y_pix < self.img_width:
#                         label[x_pix, y_pix] = 1
#         return label

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