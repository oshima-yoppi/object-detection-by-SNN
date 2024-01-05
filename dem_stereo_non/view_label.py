import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from module.const_blender import *
import os
import cv2
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int, help="numver of dem")
args = parser.parse_args()
n = args.n


raw_label_path = os.path.join(RAW_LABEL_PATH, f"{str(n).zfill(5)}.npz")

vido_path = os.path.join(VIDEO_CENTER_PATH, f"{str(n).zfill(5)}.avi")
cap = cv2.VideoCapture(vido_path)
ret, frame = cap.read()
print(ret)
print(frame.shape)
fig = plt.figure()
plt.imshow(frame)
save_path = os.path.join("result_thesis", "den_noisy_noize.pdf")
# plt.show()
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
plt.savefig(save_path)

# pp = PdfPages(save_path)
# pp.savefig(fig)
