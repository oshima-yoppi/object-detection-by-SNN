import numpy as np
import random
import math
import scipy.io
import os
import shutil
from tqdm import tqdm

def main(dem_path, save_path):
    dem = scipy.io.loadmat(dem_path)['true_DEM']
    np.save(save_path, dem)
    return 
if __name__ == "__main__":
    DATA_NUM = 3001
    SAVE_DIR = "blender/dem"

    

    for i in tqdm(range(DATA_NUM)):
        file_name = f'dem_{i}.npy'
        save_path = os.path.join(SAVE_DIR, file_name)
        dem_path = f"DEM/128pix_(0-3deg)_dem(lidar_noisy)_boulder/model/real_model_{i}.mat"
        main(dem_path, save_path)