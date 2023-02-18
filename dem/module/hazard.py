import numpy as np
import math
import cv2

class LunarHazardMapper:
    def __init__(self, shape, rough, theta, window=33):
        self.shape = shape
        self.rough = rough
        self.theta = theta
        self.window = window
    def Get_Slope(self, roi):
        middle = self.window//2
        N = roi[middle-1,middle]
        S = roi[middle+1,middle]
        E = roi[middle,middle+1]
        W = roi[middle,middle-1]
        SE = roi[middle+1,middle+1]
        NE = roi[middle-1,middle+1]
        SW = roi[middle+1,middle-1]
        NW = roi[middle-1,middle-1]
        fx = (SE-SW+np.sqrt(2)*(E-W)+NE-NW)/(4+2*np.sqrt(2))
        fy = (NW-SW+np.sqrt(2)*(N-S)+NE-SE)/(4+2*np.sqrt(2))
        theta = np.arctan(math.sqrt((fx**2+fy**2)))
        return theta
    def Get_Roughness(self, cropped):
        roughness = np.var(cropped)
        return roughness
    def map_hazard(self):
        # ウィンドウ大きさ
        half_window = self.window//2
        self.dem_padding = np.pad(self.dem, half_window)
        
        scale = 1.0

        rotate_list = [0.0] # simple label 適用時
        # rotate_list = [0.0, 45]

        S = np.zeros((self.shape,self.shape)) # slope for each pixel
        R = np.zeros((self.shape,self.shape)) # roughness for each pixel
        size = (self.window,self.window)


        for i in range(half_window, half_window+self.shape):
            for j in range(half_window, half_window+self.shape):
                cropped_window = self.dem_padding[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
                suiheido = self.Get_Slope(cropped_window)
                S[i,j] = max(S[i,j], suiheido)
                    
                heitando = self.Get_Roughness(cropped_window)
                R[i, j] = max(R[i, j], heitando)
                    
             

        # for row in range(self.window//2+1, self.shape-(self.window//2)-1, 1):
        #     for col in range(self.window//2+1, self.shape-(self.window//2)-1, 1):
        #         for angle in rotate_list:
        #             center = (int(col), int(row))
        #             #print(center)
        #             # trans = cv2.getRotationMatrix2D(center, angle, scale)
        #             # DEM2 = cv2.warpAffine(self.dem, trans, (self.shape,self.shape),cv2.INTER_CUBIC)
                   
        #             # 切り抜く。
        #             # cropped = cv2.getRectSubPix(DEM2, size, center)
        #             cropped = cv2.getRectSubPix(self.dem, size, center)
        #             suiheido = self.Get_Slope(cropped)
        #             if suiheido > S[row][col]: # ワーストケースを記録
        #                 S[row][col] = suiheido
                    
                    
        #             # 画像外枠境界線で粗さの取得を禁止する
        #             if row==self.window//2+1 or col==self.window//2+1:
        #                 heitando=0
        #             elif row==self.shape-(self.window//2)-2 or col==self.shape-(self.window//2)-2:
        #                 heitando=0
        #             else:
        #                 #heitando = Get_Roughness_alhat(cropped, m)   
        #                 heitando = self.Get_Roughness(cropped)
        #             if heitando > R[row][col]:
        #                 R[row][col] = heitando
                    

        S = S>math.radians(self.theta)
        R = R>self.rough

        hazard = (S|R)
        return hazard