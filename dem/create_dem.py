import math
import numpy as np
from module.const import*
import random
import matplotlib.pyplot as plt


class LunarDEMGeneartor:
    def __init__(self, x_pix, y_pix):
        self.dem = np.zeros((x_pix, y_pix))
    
    def put_crater(self, dem):

def put_crater(dem, n_crater):
    x_shape, y_shape = dem.shape
    eps = 1e-6
    for i in range(n_crater):
        center_x, center_y = random.uniform(-5, x_shape+5), random.uniform(-5, y_shape+5)
        

        min_lentgh_of_crater = 0.2 # [m]
        
        radius = random.uniform(min_lentgh_of_crater//METER_PER_GRID, x_shape//5)
        if radius == 0:radius =1
        print(center_x, center_y, radius)

        #    % H_r = 150 + abs(5*randn())
        H_ro = 0.036*(2*radius)**1.014
        H_r = H_ro
        H_c = 0.196*(2*radius)**1.010 - H_ro 
        W_r = 0.257*(2*radius)**1.011
        alpha = (H_c+H_r)*radius/(H_c+H_ro+eps)
        beta = radius+(1-(H_c+H_r)/(H_c+H_ro+eps))*W_r
        A = -3*radius**3 + 2*radius**2*beta + 2*radius*beta**2 + 2*beta**3
        
        for i in range(x_shape):
            for j in range(y_shape):
                h = 0
                r = math.sqrt(abs(i-center_x)**2 + abs(j-center_y)**2)

                if r <= alpha:
                    h = (H_c+H_ro)*(r**2/radius**2)-H_c;
                elif r <= radius:
                    h = ((H_c + H_ro)**2/(H_r - H_ro+eps) - H_c + eps)*((r/radius) - 1+eps)**2 + H_r
                elif r <= beta:
                    h =  (H_r*(radius+W_r)**3*A)/(W_r*beta**4*(radius-beta)**2*(3*radius**2+3*radius*W_r+W_r**2)+eps) * (r-radius)**2*(r-beta*(1+(beta**3-radius**3)/A))+H_r
                elif r <= radius+W_r:
                    h = H_r*(radius+W_r)**3/((radius+W_r)**3-radius**3+eps)*(r/radius)**(-3) - (H_r*radius**3)/((radius+W_r)**3-radius**3)
                
                dem[i,j] += h
    return dem

def put_boulder(dem, n_boulder):
    x_shape, y_shape = dem.shape
    eps = 1e-6
    for i in range(n_boulder):
        center_x, center_y = random.uniform(-5, x_shape+5), random.uniform(-5, y_shape+5)

        min_lentgh_of_boulder = 0.1 
        max_lentgh_of_boulder = 1
        x_axis = random.uniform(min_lentgh_of_boulder//METER_PER_GRID, max_lentgh_of_boulder//METER_PER_GRID) # 0.1m~1m
        y_axis = random.uniform(min_lentgh_of_boulder//METER_PER_GRID, max_lentgh_of_boulder//METER_PER_GRID)
        z_axis = max(x_axis, y_axis)*0.5 # by kariya

        
   

       
        for i in range(x_shape):
            for j in range(y_shape):
                # h = 0
                x_ = i - center_x
                y_ = j - center_y

                # 楕円方程式無いに存在するか
                # print(x_axis)
                # print(x_/x_axis)
                if y_**2 <= y_axis**2*(1-(x_/x_axis)**2):
                    dem[i,j] += generate_elllipsoid(i,j,center_x, center_y, x_axis, y_axis, z_axis)
                
              
    return dem

def generate_elllipsoid(x,y,xc,yc,xr,yr,zr):
    return zr*math.sqrt(1-((x-xc)**2/xr**2)-((y-yc)**2/yr**2))




if __name__ == "__main__":
    dem = np.zeros((100,100))
    # dem = put_crater(dem, 3)
    dem = put_boulder(dem, 5)
    plt.figure()
    plt.imshow(dem)
    plt.show()


