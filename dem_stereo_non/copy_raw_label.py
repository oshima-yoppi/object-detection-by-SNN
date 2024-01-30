import h5py
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

moto_right_dir = "../dem_stereo/raw-data/th-0.15/right"
moto_left_dir = "../dem_stereo/raw-data/th-0.15/left"
moto_right_paths = os.listdir(moto_right_dir)
moto_left_paths = os.listdir(moto_left_dir)
# copy_dir = "../dem_stereo_noisy/dataset/100000_(130,173)_th-0.15_MaxStep-10_EventCount-False_Distinguish-True_LeargeData-True"
# copy_paths = os.listdir(copy_dir)

copy_right_dir = "./raw-data/th-0.15/right"
copy_left_dir = "./raw-data/th-0.15/left"
copy_right_paths = os.listdir(copy_right_dir)
copy_left_paths = os.listdir(copy_left_dir)


for moto_left_path, moto_right_path in zip(tqdm(moto_left_paths), moto_right_paths):
    moto_left_path = os.path.join(moto_left_dir, moto_left_path)
    moto_right_path = os.path.join(moto_right_dir, moto_right_path)
    with h5py.File(moto_left_path, "r") as f:
        print(f.keys())
    break
for (
    moto_left_path,
    moto_right_path,
    copy_left_path,
    copy_right_path,
) in zip(tqdm(moto_left_paths), moto_right_paths, copy_left_paths, copy_right_paths):
    moto_left_path = os.path.join(moto_left_dir, moto_left_path)
    moto_right_path = os.path.join(moto_right_dir, moto_right_path)
    copy_left_path = os.path.join(copy_left_dir, copy_left_path)
    copy_right_path = os.path.join(copy_right_dir, copy_right_path)
    with h5py.File(moto_left_path, "r") as f:
        left_label = f["label"][:]
    with h5py.File(moto_right_path, "r") as f:
        right_label = f["label"][:]
    with h5py.File(copy_right_path, "r+") as f:
        if "label" in f:
            del f["label"]
        f.create_dataset("label", data=right_label)
    with h5py.File(copy_left_path, "r+") as f:
        if "label" in f:
            del f["label"]
        f.create_dataset("label", data=left_label)
