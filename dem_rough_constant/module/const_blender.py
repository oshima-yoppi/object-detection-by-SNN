# blender関連の定数
FOCAL = 0.050  # 焦点距離
IMG_HEIGHT, IMG_WIDTH = 260, 346  # カメラの大きさ[pix]
SENSOR_HEIGHT, SENSOR_WIDTH = 0.026, 0.0346  # イメージセンサの大きさ [m]
CAM_X, CAM_Y, CAM_Z = 2.4, 2.4, 6  # カメラの初期位置[m,m,m]
METER_PER_GRID = 0.018  # 1グリッドの大きさ[m]


# path
VIDEO_PATH = "blender/video"
VIDEO_ONLY_BOULDER_PATH = "blender/video_only_boulder"
DEM_NP_PATH = "blender/dem"
DEM_ONLY_BOULDER_PATH = "blender/dem_only_boulder"

# https://docs.blender.org/manual/en/2.79/data_system/files/relative_paths.html
VIDEO_PATH_BLENDER = "//" + VIDEO_PATH
VIDEO_ONLY_BOULDER_PATH_BLENDER = "//" + VIDEO_ONLY_BOULDER_PATH
DEM_NP_PATH_BLENDER = "//" + DEM_NP_PATH
DEM_ONLY_BOULDER_PATH_BLENDER = "//" + DEM_ONLY_BOULDER_PATH
