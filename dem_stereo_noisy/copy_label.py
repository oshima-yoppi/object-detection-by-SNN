import h5py
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

moto_dir = "../dem_stereo_non/dataset/100000_(130,173)_th-0.15_MaxStep-10_EventCount-False_Distinguish-True_LeargeData-True"
moto_paths = os.listdir(moto_dir)

copy_dir = "../dem_stereo_noisy/dataset/100000_(130,173)_th-0.15_MaxStep-10_EventCount-False_Distinguish-True_LeargeData-True"
copy_paths = os.listdir(copy_dir)
for moto_path, copy_path in zip(tqdm(moto_paths), copy_paths):
    moto_path = os.path.join(moto_dir, moto_path)
    copy_path = os.path.join(copy_dir, copy_path)
    # print(moto_path)
    with h5py.File(moto_path, "r") as f:
        moto_label = f["label"][:]
        moto_label_fine = f["label_fine"][:]
    # copy 先にmotoファイルデータをコピーしてラベルを上書きする
    with h5py.File(copy_path, "r+") as f:
        f["label"][:] = moto_label
        f["label_fine"][:] = moto_label_fine
    # with h5py.File(copy_path, "r") as f:
    #     copy_label = f["label"][:]
    #     copy_label_fine = f["label_fine"][:]

    # plt.subplot(2, 2, 1)
    # plt.imshow(moto_label[0])
    # plt.subplot(2, 2, 2)
    # plt.imshow(copy_label[0])
    # plt.subplot(2, 2, 3)
    # plt.imshow(moto_label_fine[0])
    # plt.subplot(2, 2, 4)
    # plt.imshow(copy_label_fine[0])
    # plt.show()
    # # with h5py.File(copy_path, "r+") as f:
    # #     f["label"][:] = moto_label
    # #     f["label_fine"][:] = moto_label_fine
    # # # break
