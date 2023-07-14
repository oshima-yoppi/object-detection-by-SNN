import glob
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


new_dir_path = (
    r"C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(0deg)_dem(noisy)_ver2/kuga_label"
)
os.makedirs(new_dir_path, exist_ok=True)
original_DEM_path = (
    r"C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(0deg)_dem(noisy)_ver2/model/"
)


file_mei = 11
add_path_ = "observed_model_" + str(file_mei) + ".mat"
read_path_ = os.path.join(original_DEM_path, add_path_)


rei = scipy.io.loadmat(read_path_)["DEM"]
print(rei)
height = rei.shape[0]
width = rei.shape[1]
# ウィンドウ大きさ
F = 8


def Get_Slope(h, w, D, mu, sigma):

    roi = D[(h - F // 2) : (h + F // 2), (w - F // 2) : (w + F // 2)]
    mu_roi = np.mean(roi)
    suiheido = (mu_roi - mu) / sigma
    return abs(suiheido)


def Get_Roughness(h, w, D):
    roi = D[(h - F // 2) : (h + F // 2), (w - F // 2) : (w + F // 2)]

    s_roi = np.std(roi)
    heitando = s_roi / sigma
    return heitando


# print(height,width)
V = np.zeros((height, width))  # safety for each pixel
S = np.zeros((height, width))  # slope for each pixel
R = np.zeros((height, width))  # roughness for each pixel


for file_num in range(10000, 16640):
    # original_DEM_path = r'C:/Users/aki/Documents/GitHub/deep/DEM/64pix_(0deg)_dem/model/'
    add_path = "observed_model_" + str(file_num) + ".mat"
    file = os.path.join(original_DEM_path, add_path)
    print("READ_PATH:", file)

    print("input : ", file)
    DEM = scipy.io.loadmat(file)["DEM"]
    DEM = np.array(DEM, dtype="float32")
    mu = np.mean(DEM)
    sigma = np.std(DEM)
    # print("heikin ",mu)

    # laplacian =  cv2.Laplacian(DEM,cv2.CV_64F)
    V = np.zeros((height, width))  # safety for each pixel
    S = np.zeros((height, width))  # slope for each pixel
    R = np.zeros((height, width))  # roughness for each pixel

    for row in range(F // 2, height - (F // 2), 1):
        for col in range(F // 2, width - (F // 2), 1):
            s = Get_Slope(row, col, DEM, mu, sigma)
            r = Get_Roughness(row, col, DEM)
            if s > S[row][col]:
                S[row][col] = s
            if r > R[row][col]:
                R[row][col] = r

    Vthm = np.mean(S)
    Vths = np.mean(R)
    S = S > Vthm
    R = R > 1.5 * Vths
    hazard = S | R

    save_path = new_dir_path + "/label_" + str(file_num) + ".mat"
    print("SAVE_PATH:", save_path)
    scipy.io.savemat(save_path, {"label_data": hazard})

    # np.set_printoptions(threshold=np.inf)
    # print(S+R)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax1.set_title("original")
    ax1.imshow(DEM)
    ax2.set_title("slope")
    ax2.imshow(S, cmap="jet")
    ax3.set_title("roughness")
    ax3.imshow(R, cmap="jet")

    hazard = S | R

    save_path = new_dir_path + "/label_" + str(file_num) + ".mat"
    print("SAVE_PATH:", save_path)
    scipy.io.savemat(save_path, {"label_data": hazard})

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_title("hazard_thr")
    ax4.imshow(hazard)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_title("slope_thr")
    ax5.imshow(S)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_title("roughness_thr")
    ax6.imshow(R)
    save_path = new_dir_path + "/label_" + str(file_num) + ".jpg"
    plt.savefig(save_path)
