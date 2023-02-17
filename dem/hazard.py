import glob
import scipy.io
import numpy as  np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2
import os
from tqdm import tqdm
import h5py
import shutil
import time
#

from DEM.ransac import *
from module.const import *
from module import convert_label, cmd
from module.const import *



def Get_Slope(roi):
    W = roi[0,2]
    E = roi[4,2]
    S = roi[2,4]
    N = roi[2,0]
    SE = roi[4,4]
    SW = roi[0,4]
    NE = roi[4,0]
    NW = roi[0,0]

    fx = (SE-SW+np.sqrt(2)*(E-W)+NE-NW)/(4+2*np.sqrt(2))
    fy = (NW-SW+np.sqrt(2)*(N-S)+NE-SE)/(4+2*np.sqrt(2))
    theta = np.arctan(math.sqrt((fx**2+fy**2)))
    return theta 

    

def Get_Roughness(cropped):
    roughness = np.var(cropped)
    return roughness

def make_hazard(DEM):
    # ウィンドウ大きさ
    F = 5
    height = DEM.shape[0]
    width = DEM.shape[1]


    DEM = np.array(DEM, dtype='float32')


    scale = 1.0

    # rotate_list = [0.0] # simple label 適用時
    rotate_list = [0.0, 45]

    S = np.zeros((height,width)) # slope for each pixel
    R = np.zeros((height,width)) # roughness for each pixel
    size = (F,F)
    for row in range(F//2+1, height-(F//2)-1, 1):
        for col in range(F//2+1, width-(F//2)-1, 1):
            for angle in rotate_list:
                center = (int(col), int(row))
                #print(center)
                trans = cv2.getRotationMatrix2D(center, angle, scale)
                DEM2 = cv2.warpAffine(DEM, trans, (width,height),cv2.INTER_CUBIC)
            
                #roi = DEM2[(row-F//2):(row+F//2),(col-F//2):(col+F//2)]
                # 切り抜く。
                cropped = cv2.getRectSubPix(DEM2, size, center)

                
                suiheido = Get_Slope(cropped)
                if suiheido > S[row][col]: # ワーストケースを記録
                    S[row][col] = suiheido
                   
                
                # 画像外枠境界線で粗さの取得を禁止する
                if row==F//2+1 or col==F//2+1:
                    heitando=0
                elif row==height-(F//2)-2 or col==width-(F//2)-2:
                    heitando=0
                else:
                    #heitando = Get_Roughness_alhat(cropped, m)   
                    heitando = Get_Roughness(cropped)
                if heitando > R[row][col]:
                    R[row][col] = heitando
                

    S = S>0.6
    R = R>0.1

    hazard = (S|R)
    return hazard



if __name__ == '__main__':

    if os.path.exists(RAW_EVENT_PATH):
        shutil.rmtree(RAW_EVENT_PATH)
    os.mkdir(RAW_EVENT_PATH)


    data_num = 3001
    converter = convert_label.Dem2Img(focal=FOCAL, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, sensor_heitght=SENSOR_HEIGHT,
    sensor_width=SENSOR_WIDTH, cam_x=CAM_X, cam_y=CAM_Y, cam_z=CAM_Z)

    for file_num in tqdm(range( data_num)):
        dem_path = f"blender/dem/dem_{file_num}.npy"
        DEM = np.load(dem_path)     
        hazard = make_hazard(DEM)


        hazard_reshape = converter(hazard)
        # fig = plt.figure()
        # ax1 = fig.add_subplot(121)
        # ax2 = fig.add_subplot(122)
        # ax1.imshow(DEM)
        # ax2.imshow(hazard_reshape)
        # plt.show()

        savefile_path = f'{str(file_num).zfill(5)}.h5'
        savefile_path = os.path.join(RAW_EVENT_PATH, savefile_path)
        with h5py.File(savefile_path, 'a') as f:
            f.create_dataset("label", data=hazard_reshape)
        
        cmd.v2e_cmd(file_num)
        

        

        
        

        
