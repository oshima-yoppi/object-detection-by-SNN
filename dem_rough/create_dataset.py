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

# from DEM.ransac import *
from module.const import *
from module import convert_label, cmd
from module.const import *
# from sklearn.preprocessing import MinMaxScaler




if __name__ == '__main__':
    
    if os.path.exists(RAW_EVENT_PATH):
        shutil.rmtree(RAW_EVENT_PATH)
    os.makedirs(RAW_EVENT_PATH)


    data_num = 3000 
    label_path = LABEL_PATH

    for file_num in tqdm(range( data_num)):
        num_zip = str(file_num).zfill(5)
        label_path = os.path.join(LABEL_PATH, f"{num_zip}.npy")
        label = np.load(label_path)
        # plt.figure()
        # plt.imshow(label)
        # plt.show()
        


        savefile_path = f'{str(file_num).zfill(5)}.h5'
        savefile_path = os.path.join(RAW_EVENT_PATH, savefile_path)
        with h5py.File(savefile_path, 'a') as f:
            f.create_dataset("label", data=label)
        
        cmd.v2e_cmd(file_num, EVENT_TH)
        

        
        

        
