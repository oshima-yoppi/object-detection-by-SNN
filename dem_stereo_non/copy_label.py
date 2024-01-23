import h5py
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

moto_dir = "../dem_stereo/dataset/100000_(130,173)_th-0.15_MaxStep-10_EventCount-False_Distinguish-True_LeargeData-True"
moto_paths = os.listdir(moto_dir)

copy_dir = "../dem_stereo_non/dataset/100000_(130,173)_th-0.15_MaxStep-10_EventCount-False_Distinguish-True_LeargeData-True"
copy_paths = os.listdir(copy_dir)
for moto_path, copy_path in zip(tqdm(moto_paths), copy_paths):
    moto_path = os.path.join(moto_dir, moto_path)
    copy_path = os.path.join(copy_dir, copy_path)
    # print(moto_path)
    with h5py.File(moto_path, "r") as f:
        moto_label = f["label"][:]
        moto_label_fine = f["label_fine"][:]
    # print(moto_label.shape, moto_label_fine.shape)
    # copy 先にmotoファイルデータをコピーしてラベルを上書きする
    with h5py.File(copy_path, "r+") as f:
        if "label" in f:
            del f["label"]
        if "label_fine" in f:
            del f["label_fine"]
        # del f["label"]
        # del f["label_fine"]
        f.create_dataset("label", data=moto_label)
        f.create_dataset("label_fine", data=moto_label_fine)
