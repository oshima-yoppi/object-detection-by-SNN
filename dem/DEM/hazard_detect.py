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
#


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz

def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def plot_plane(a, b, c, d):
    xx, yy = np.mgrid[:10, :10]
    smooth=1e-6
    return xx, yy, (-d - a * xx - b * yy) / (c+smooth)

def Get_Slope_alhat(roi):

    n = 100
    max_iterations = 100
    goal_inliers = n * 0.1
    F = 5
    xyzs = np.zeros((F**2, 3))
    i = 0
    for x in range(F):
        for y in range(F):
    
            z = roi[x][y]
            #print(i,x,y,z)
            xyzs[i][0] = x 
            xyzs[i][1] = y
            xyzs[i][2] = z
            i += 1
    # RANSAC
    m, best_inliners = run_ransac(xyzs, estimate, lambda x, y: is_inlier(x, y, 0.001), 3, goal_inliers, max_iterations)
    a, b, c, d = m
    # 描画
    xx, yy, zz = plot_plane(a, b, c, d)

    
    center = roi[2,2]
    W = roi[0,2]
    E = roi[4,2]
    S = roi[2,4]
    N = roi[2,0]
    SE = roi[4,4]
    SW = roi[0,4]
    NE = roi[4,0]
    NW = roi[0,0]
    fx = ((SE-SW+np.sqrt(2))*(E-W)+NE-NW)/(4+2*np.sqrt(2))
    fy = ((NW-SW+np.sqrt(2))*(N-S)+NE-SE)/(4+2*np.sqrt(2))
    theta = np.arctan(fx**2+fy**2)
    """
    costheta = np.abs(c)/np.sqrt(((a)**2+(b)**2+(c)**2))
    #print("costheta:c:np.sqrt",costheta,c,np.sqrt(((a)**2+(b)**2+(c)**2)))
    theta = 1-np.arccosh(costheta)
    """
    return  theta, m

def Get_Roughness_alhat(cropped, m):
    F = 5
    a,b,c,d = m
    smooth = 1e-6
    diff = [0]
    for x in range(F):
        for y in range(F):
            z = (-a/(c+smooth))*x + (-b/(c+smooth))*y + (-d/(c+smooth))
            #print("z",z)
            #print("cropped[x][y]-z",cropped[x][y]-z)
            diff_ = cropped[x][y]-z
         
            if diff_ > 100:
                pass
            else:
                diff.append(cropped[x][y]-z)
    
    roughness = max(diff)
    return roughness





def Get_Slope(roi):

    n = 100
    max_iterations = 10
    goal_inliers = n * 0.1
    F = 5

    xyzs = np.zeros((F**2, 3))
    smooth = 1e-6
    center = roi[2,2]
    W = roi[0,2]
    E = roi[4,2]
    S = roi[2,4]
    N = roi[2,0]
    SE = roi[4,4]
    SW = roi[0,4]
    NE = roi[4,0]
    NW = roi[0,0]
    fx = ((SE-SW+np.sqrt(2))*(E-W)+NE-NW)/(4+2*np.sqrt(2))
    fy = ((NW-SW+np.sqrt(2))*(N-S)+NE-SE)/(4+2*np.sqrt(2))
    theta = np.arctan(fx**2+fy**2)


    """
    #ロバスト平面の方程式から
    costheta = np.abs(c)/np.sqrt(((a)**2+(b)**2+(c)**2))
    #print("costheta:c:np.sqrt",costheta,c,np.sqrt(((a)**2+(b)**2+(c)**2)))
    theta = 1-np.arccosh(costheta)
    
    
    #print('SLOPE:a:b:c:d',ans,a,b,c,d)
    #plt.show()
    """
    #一枚当たりの処理時間を表示

    return theta #, m

    

def Get_Roughness(cropped):

    #a,b,c,d = m
    smooth = 1e-6
    """
    #print("cropped.shape",cropped.shape,type(cropped))
    #print("x",x.shape)

    z = x_ary*(-a/(c+smooth)) + y_ary*(-b/(c+smooth))+ (-d/(c+smooth))
    #print("z",z.shape)
    diff = cropped-z
    #print("diff",diff.shape)
    roughness = np.nanmax(diff) 
    
    diff = [0]
    for x in range(F):
        for y in range(F):
            z = (-a/(c+smooth))*x + (-b/(c+smooth))*y + (-d/(c+smooth))
            #print("z",z)
            #print("cropped[x][y]-z",cropped[x][y]-z)
            diff_ = cropped[x][y]-z
            if diff_ > 1:
                pass
            else:
                diff.append(cropped[x][y]-z)

    roughness = max(diff) 
    """
    roughness = np.var(cropped)
    return roughness

# Get_roughness内の計算用配列
#x_ary = np.array([range(i,i+8) for i in [0,1,2,3,4,5,6,7]])
#y_ary = np.array([range(i,i+8) for i in [0,1,2,3,4,5,6,7]])
if __name__ == '__main__':
    #new_dir_path = r'G:\マイドライブ\DEM\64pix_dem(lidar_noisy)_boulder_evaluate\model\simple_label'
    new_dir_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0-3deg)_dem(lidar_noisy)_boulder\simple_label'
    #new_dir_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0-3deg)_dem(lidar_noisy)\simple_label'
    #new_dir_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0deg)_dem(noisy)_evaluate_112/simple_label'
    #new_dir_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0deg)_dem(noisy)_evaluate_112/alhat_label'
    #new_dir_path = r'C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(0-5deg)_dem(noisy)/simple_label'
    #new_dir_path = r'C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(0deg)_dem(noisy)_ver2/alhat_label'
    #new_dir_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0deg)_dem(noisy)_ver2/simple_label'
    #new_dir_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0deg)_dem(noisy)_evaluate/simple_label'
    #new_dir_path = r'C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(0deg)_dem(noisy)_evaluate\simple_label_quantum'
    os.makedirs(new_dir_path, exist_ok=True)
    #original_DEM_path = r'G:\マイドライブ\DEM\64pix_dem(lidar_noisy)_boulder_evaluate\model'
    original_DEM_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0-3deg)_dem(lidar_noisy)_boulder\model'
    #original_DEM_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0-3deg)_dem(lidar_noisy)\model(t-10)'
    #original_DEM_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(0deg)_dem(noisy)_evaluate_112\noise_0\model'
    #original_DEM_path = r'C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(0-5deg)_dem(noisy)/model/'
    #original_DEM_path = r'C:\Users\aki\Documents\GitHub\deep\DEM\64pix_(deg)_dem(noisy)_ver2/model/'
    #original_DEM_path = r'C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(0deg)_dem(noisy)_evaluate\model'
    file_mei = 1
    observed = 0
    lidar_noised = 0

    if lidar_noised:
        add_path_ = 'lidar_noised_model_'+str(file_mei)+'.mat'
        read_path_ = os.path.join(original_DEM_path,add_path_)
        rei = scipy.io.loadmat(read_path_)['Lidar_noised_DEM']
    elif observed:
        add_path_ = 'observed_model_'+str(file_mei)+'.mat'
        read_path_ = os.path.join(original_DEM_path,add_path_)
        rei = scipy.io.loadmat(read_path_)['DEM']
    else:
        add_path_ = 'real_model_'+str(file_mei)+'.mat'
        read_path_ = os.path.join(original_DEM_path,add_path_)
        rei = scipy.io.loadmat(read_path_)['true_DEM']
    read_path_ = os.path.join(original_DEM_path,add_path_)



    print(rei)
    height = rei.shape[0]
    width = rei.shape[1]
    # ウィンドウ大きさ
    F = 5
    for file_num in range(205,20000):
        if lidar_noised:
            add_path_ = 'lidar_noised_model_'+str(file_num)+'.mat'
            read_path_ = os.path.join(original_DEM_path,add_path_)
            DEM = scipy.io.loadmat(read_path_)['Lidar_noised_DEM']
        elif observed:
            add_path_ = 'observed_model_'+str(file_num)+'.mat'
            read_path_ = os.path.join(original_DEM_path,add_path_)
            DEM = scipy.io.loadmat(read_path_)['DEM']
        else:
            add_path_ = 'real_model_'+str(file_num)+'.mat'
            read_path_ = os.path.join(original_DEM_path,add_path_)
            DEM = scipy.io.loadmat(read_path_)['true_DEM'] 
        

        print('READ_PATH:',read_path_)
        start = time.time()


        #print(DEM.dtype)
        DEM = np.array(DEM, dtype='float32')
        mu = np.mean(DEM)
        sigma = np.std(DEM)

        scale = 1.0

        #rotate_list = [0.0] # simple label 適用時
        rotate_list = [0.0,30.0,60.0,90.0,120.0,150.0,180.0]# ALHAT 適用時
        #rotate_list = [0.0,90.0,180.0,270.0,360.0]

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

                    """
                    fig = plt.figure()
                    ax1 = fig.add_subplot(1,3,1)
                    ax2 = fig.add_subplot(1,3,2)
                    ax3 = fig.add_subplot(1,3,3)
                    ax1.imshow(DEM)
                    ax2.imshow(DEM2)
                    ax3.imshow(cropped)
                    plt.show()
                    
                    """
                   # suiheido, m = Get_Slope_alhat(cropped)
                    suiheido = Get_Slope(cropped)
                    if suiheido > S[row][col]: # ワーストケースを記録
                        S[row][col] = suiheido
                        #print("suiheido",suiheido)
                    
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
                    

        #S = min_max(S)
        #R = min_max(R)
        #np.set_printoptions(threshold=np.inf)
        #print("S:",S)
        #print("R:",R)
        fig = plt.figure()
        ax1 = fig.add_subplot(2,3,1)
        ax2 = fig.add_subplot(2,3,2)
        ax3 = fig.add_subplot(2,3,3)
        ax1.set_title('original')
        ax1.imshow(DEM)
        ax2.set_title('slope'+str(np.mean(S)))
        ax2.imshow(S,cmap='jet') 
        ax3.set_title('roughness'+str(np.mean(R)))
        ax3.imshow(R,cmap='jet')
        

        '''
        Vthm = np.mean(S)
        Vths = np.mean(R)
        S = S>1.5*Vthm # SLOPE
        print("1.5*Vthm",1.5*Vthm)
        R = R>1.5*Vths # ROUGHNESS
        print("1.5*Vths",1.5*Vths)
        '''
        #print("max S",np.max(S))
        #print("max R",np.max(R))
        #S = S>0.98
        S = S>0.6
        R = R>0.1

        hazard = (S|R)

        save_path = new_dir_path + '/label_'+ str(file_num) + '.mat'
        print('SAVE_PATH:',save_path)
        scipy.io.savemat(save_path, {'label_data':hazard})

        ax4 = fig.add_subplot(2,3,4)
        ax4.set_title('hazard_thr')
        ax4.imshow(hazard)
        ax5 = fig.add_subplot(2,3,5)
        ax5.set_title('slope_thr')
        ax5.imshow(S)
        ax6 = fig.add_subplot(2,3,6)
        ax6.set_title('roughness_thr')
        ax6.imshow(R)
        save_path = new_dir_path + '/label_'+ str(file_num) + '.jpg'
        plt.savefig(save_path)

        #一枚当たりの処理時間を表示
        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        

        
