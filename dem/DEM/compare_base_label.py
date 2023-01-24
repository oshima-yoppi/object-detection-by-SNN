import glob
import scipy.io
import numpy as  np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import cv2
import os
from ransac import *
from sklearn.preprocessing import MinMaxScaler
import time
from hazard_detect import Get_Roughness_alhat, Get_Slope_alhat
import csv
# ベースラベル　IoU=100% シンプルラベルに固定
# base_label_path = r'C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(0-5deg)_dem(noisy)/simple_label'
# base_label_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0deg)_dem(noisy)_evaluate_112/alhat_label'
# base_label_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0-3deg)_dem(lidar_noisy)_boulder\simple_label'
base_label_path = r'G:\マイドライブ\DEM\64pix_dem(lidar_noisy)_boulder_evaluate\simple_label'
# ターゲット DEM_path SNNの比較対象
#target_DEM_path = r'C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(0-5deg)_dem(noisy)/model(t-100)'
#target_DEM_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\simple_crater\model'
#target_DEM_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0deg)_dem(noisy)_evaluate_112\noise_0\model'
#target_DEM_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0-3deg)_dem(lidar_noisy)_boulder\model(t-100)'
target_DEM_path = r'G:\マイドライブ\DEM\64pix_dem(lidar_noisy)_boulder_evaluate\model(t-100)' #1/26
print('START')

def iou_score(outputs, labels):
    smooth = 1e-6
    #print('output shape',outputs.shape)

    #print("outputs : ",outputs)
    iou = []
    precision = []
    recall =[]
    cnt = []

    output = np.where(outputs>0,1,0)
    label = np.where(labels>0,1,0)
    intersection = (np.uint64(output) & np.uint64(label)).sum((0,1)) # will be zero if Trueth=0 or Prediction=0
    union = (np.uint64(output) | np.uint64(label)).sum((0,1)) # will be zero if both are 0
    
    precision.append(intersection / np.uint64(output).sum((0,1)))
    recall.append(intersection / np.uint64(label).sum((0,1)))
    iou.append((intersection + smooth) / (union + smooth))

    return iou, precision, recall



for file_num in range(367,1000): #14593,14594,1



    # ベースラベル読み込み
    add_path = 'label_'+str(file_num)+'.mat' 
    read_path = os.path.join(base_label_path,add_path)
    base_label = scipy.io.loadmat(read_path)['label_data']

    """
    # ターゲット　DEM（量子化後)読み込み
    add_path = 'observed_model_'+str(file_num)+'.mat' # noisy
    read_path = os.path.join(target_DEM_path,add_path)
    DEM = scipy.io.loadmat(read_path)['DEM']
    #DEM = scipy.io.loadmat(read_path)['model']
    
    # ターゲット DEM(true_DEM)読み込み
    add_path = add_path_ = 'real_model_'+str(file_num)+'.mat'
    read_path = os.path.join(target_DEM_path,add_path)
    DEM = scipy.io.loadmat(read_path)['true_DEM']

    """
    # ターゲット DEM(lidar_noised_DEM)読み込み
    add_path_ = 'observed_model_'+str(file_num)+'.mat'
    read_path_ = os.path.join(target_DEM_path,add_path_)
    DEM = scipy.io.loadmat(read_path_)['DEM']
    
    DEM = np.array(DEM, dtype='float32')
    mu = np.mean(DEM)
    sigma = np.std(DEM)
    height = DEM.shape[0]
    width = DEM.shape[1]
    # ウィンドウ大きさ
    F = 5
    scale = 1.0

    #rotate_list = [0.0,30.0,60.0,90.0,120.0,150.0,180.0,210.0,240.0,270.0,300.0,330.0,360.0]
    rotate_list = [0.0,30.0,60.0, 90.0,120.0,150.0, 180.0]
    V = np.zeros((height,width)) # safety for each pixel
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
                suiheido, m = Get_Slope_alhat(cropped)

                if suiheido > S[row][col]: # ワーストケースを記録
                    S[row][col] = suiheido
                    #print("suiheido",suiheido)
                
                # 画像外枠境界線で粗さの取得を禁止する
                if row==F//2+1 or col==F//2+1:
                    heitando=0
                elif row==height-(F//2)-2 or col==width-(F//2)-2:
                    heitando=0
                else:
                    heitando = Get_Roughness_alhat(cropped, m)   

                if heitando > R[row][col]:
                    R[row][col] = heitando

    fig = plt.figure()
    ax1 = fig.add_subplot(2,4,1)
    ax2 = fig.add_subplot(2,4,2)
    ax3 = fig.add_subplot(2,4,3)
    ax1.set_title('DEM')
    ax1.imshow(DEM,cmap='gray')
    ax2.set_title('slope')
    ax2.imshow(S,cmap='jet') 
    ax3.set_title('roughness')
    ax3.imshow(R,cmap='jet')
    
    #S = S>0.98
    S = S>0.6

    # 2022 1/20追記
    iou_survey = 0
    if iou_survey:
     
        iou_list = []
        precision_list = []
        recall_list = []

        for i in range(100):
            i = i/100
            R_ = R>i

            hazard = (S|R_)
            iou, precision, recall = iou_score(hazard, base_label)
            #print('=======iou:',iou[0])

            iou_list.append(iou[0])
            precision_list.append(precision[0])
            recall_list.append(recall[0])

        #print(iou_list)
        max_iou_val = max(iou_list)
        max_iou_index = iou_list.index(max_iou_val)
        R = R>max_iou_index/100
    else:
        R = R>0.5


    hazard = (S|R)
    iou, precision, recall = iou_score(hazard, base_label)
    print('path : IoU : precision : recall ', str(file_num),iou, precision, recall)

    ax4 = fig.add_subplot(2,4,4)
    ax4.set_title('hazard_thr')
    ax4.imshow(hazard)
    ax5 = fig.add_subplot(2,4,6)
    ax5.set_title('slope_thr')
    ax5.imshow(S)
    ax6 = fig.add_subplot(2,4,7)
    ax6.set_title('roughness_thr')
    ax6.imshow(R)
    ax7 = fig.add_subplot(2,4,8)
    ax7.set_title('baseline_label')
    ax7.imshow(base_label)
    if iou_survey:
        ax5 = fig.add_subplot(2,4,5)
        ax5.plot(iou_list, label='iou')
        ax5.legend()

    save_path = target_DEM_path + '/alhat_label_'+ str(file_num) + '.jpg'
    plt.savefig(save_path)

    # CSV write
    filename = target_DEM_path +'alhat_simple_ioudata.csv'
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([save_path,str(iou),str(precision),str(recall)])

    if iou_survey:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(iou_list, label='iou')
        ax.plot(precision_list, label='precision')
        ax.plot(recall_list, label='recall')
        ax.legend()
        plt.show()