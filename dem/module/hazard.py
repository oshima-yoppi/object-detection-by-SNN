import numpy as np
import math
import cv2

class LunarHazardMapper:
    def __init__(self, shape, rough, theta):
        self.shape = shape
        self.rough = rough
        self.theta = theta
        self.dem = np.zeros((shape,shape))
    def Get_Slope(self, roi):
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
    def Get_Roughness(self, cropped):
        roughness = np.var(cropped)
        return roughness
    def map_hazard(self):
        # ウィンドウ大きさ
        F = 5
        scale = 1.0

        # rotate_list = [0.0] # simple label 適用時
        rotate_list = [0.0, 45]

        S = np.zeros((self.shape,self.shape)) # slope for each pixel
        R = np.zeros((self.shape,self.shape)) # roughness for each pixel
        size = (F,F)
        for row in range(F//2+1, self.shape-(F//2)-1, 1):
            for col in range(F//2+1, self.shape-(F//2)-1, 1):
                for angle in rotate_list:
                    center = (int(col), int(row))
                    #print(center)
                    trans = cv2.getRotationMatrix2D(center, angle, scale)
                    DEM2 = cv2.warpAffine(self.dem, trans, (self.shape,self.shape),cv2.INTER_CUBIC)
                    #roi = DEM2[(row-F//2):(row+F//2),(col-F//2):(col+F//2)]
                    # 切り抜く。
                    cropped = cv2.getRectSubPix(DEM2, size, center)
                    suiheido = self.Get_Slope(cropped)
                    if suiheido > S[row][col]: # ワーストケースを記録
                        S[row][col] = suiheido
                    
                    
                    # 画像外枠境界線で粗さの取得を禁止する
                    if row==F//2+1 or col==F//2+1:
                        heitando=0
                    elif row==self.shape-(F//2)-2 or col==self.shape-(F//2)-2:
                        heitando=0
                    else:
                        #heitando = Get_Roughness_alhat(cropped, m)   
                        heitando = self.Get_Roughness(cropped)
                    if heitando > R[row][col]:
                        R[row][col] = heitando
                    

        S = S>math.radians(self.theta)
        R = R>self.rough

        hazard = (S|R)
        return hazard