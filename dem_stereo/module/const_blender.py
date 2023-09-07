# blender関連の定数
FOCAL = 0.050  # 焦点距離
IMG_HEIGHT, IMG_WIDTH = 260, 346  # カメラの大きさ[pix]
SENSOR_HEIGHT, SENSOR_WIDTH = 0.024, 0.036  # イメージセンサの大きさ [m]
SENSOR_HEIGHT = SENSOR_WIDTH * IMG_HEIGHT / IMG_WIDTH  # センサの縦横比を維持するために必要
CAM_X, CAM_Y, CAM_Z = 2.4, 2.4, 6  # カメラの初期位置[m,m,m]
METER_PER_GRID = 0.018  # 1グリッドの大きさ[m]
CAMERA_THETA = 39.6  # カメラの視野角[deg](これはおそらくブレンダー側が自動で計算してくれるはず。その値をここにかく。)
DISTANCE_BETWEEN_CAMERA = 2  # 左右のカメラの間の距離[m]
OVERLAP_LENGTH = 0.4  # 二つのカメラが見る範囲をどれくらい被らせるか[m](0.3m で高度1mの誤差、0.4m で高度1.5mの誤差、0.5m で高度2mの誤差、が許容できる)


# path
VIDEO_RIGHT_PATH = "blender/video/right"
VIDEO_LEFT_PATH = "blender/video/left"
VIDEO_CENTER_PATH = "blender/video/center"


DEM_NP_PATH = "blender/dem"
LABEL_RIGHT_PATH = "blender/label/right"
LABEL_LEFT_PATH = "blender/label/left"
LABEL_CENTER_PATH = "blender/label/center"
DEM_ONLY_BOULDER_PATH = "blender/dem_only_boulder"

# https://docs.blender.org/manual/en/2.79/data_system/files/relative_paths.html
VIDEO_RIGHT_PATH_BLENDER = "//" + VIDEO_RIGHT_PATH
VIDEO_LEFT_PATH_BLENDER = "//" + VIDEO_LEFT_PATH
VIDEO_CENTER_PATH_BLENDER = "//" + VIDEO_CENTER_PATH
DEM_NP_PATH_BLENDER = "//" + DEM_NP_PATH
LABEL_RIGHT_PATH_BLENDER = "//" + LABEL_RIGHT_PATH
LABEL_LEFT_PATH_BLENDER = "//" + LABEL_LEFT_PATH
LABEL_CENTER_PATH_BLENDER = "//" + LABEL_CENTER_PATH

DEM_ONLY_BOULDER_PATH_BLENDER = "//" + DEM_ONLY_BOULDER_PATH
