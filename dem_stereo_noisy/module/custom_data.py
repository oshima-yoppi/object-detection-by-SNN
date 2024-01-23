from tarfile import DIRTYPE
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

# from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import sys
from tqdm import tqdm
import pandas as pd
import argparse
import h5py
import glob
import cv2
import numpy as np
import tonic
import tonic.transforms as transforms
from torchvision import transforms as transforms_
import torchvision.transforms as T
from tqdm import tqdm
import shutil

# from .module import const
from .const import *

# import const


class num2torch:
    def __init__(self):
        return

    def __call__(self, arr):
        return torch.from_numpy(arr)


class Number2one:
    def __init__(self):
        return

    def __call__(self, arr):
        return torch.where(arr >= 1, 1, 0)


class Fill0_Tensor:
    def __init__(self, true_shape):
        self.true_shape = true_shape

    def __call__(self, arr):
        shape_time = arr.shape[0]
        if arr.shape == self.true_shape:
            return arr
        arr_reshape = torch.zeros(self.true_shape)
        arr_reshape[:shape_time] = arr_reshape[:shape_time] + arr
        if not BOOL_DISTINGUISH_EVENT:
            arr_reshape[:, 0] = arr_reshape[:, 0] + arr_reshape[:, 1]
            arr_reshape = arr_reshape[:, 0].unsqueeze(1)
            arr_reshape = torch.where(arr_reshape >= 1, 1, 0)
            # print(arr_reshape.shape)
            # exit()
            # arr_reshape

        return arr_reshape


class ToRoughSegmentation:
    def __init__(self, rough_pix):
        self.rough_pix = rough_pix
        return

    def __call__(self, arr):
        # arr = torch.from_numpy(arr)
        # print(arr.shape, INPUT_HEIGHT, INPUT_WIDTH)
        # print(arr.shape)
        rough_label = torch.zeros((MAX_STEP, 1, self.rough_pix, self.rough_pix))
        for t in range(MAX_STEP):
            splited_label = arr[t]
            for i in range(self.rough_pix):
                for j in range(self.rough_pix):
                    splited_label = arr[
                        0,
                        i
                        * INPUT_HEIGHT
                        // self.rough_pix : (i + 1)
                        * INPUT_HEIGHT
                        // self.rough_pix,
                        j
                        * INPUT_WIDTH
                        // self.rough_pix : (j + 1)
                        * INPUT_WIDTH
                        // self.rough_pix,
                    ]
                    contein_one = torch.any(splited_label == 1)
                    if contein_one:
                        rough_label[t, 0, i, j] = 1

        return rough_label


def get_until_finishtime(raw_events, finish_time):
    for i in range(len(raw_events)):
        # print(raw_events[i], finish_time)
        if raw_events[i, 0] > finish_time:
            break
    return raw_events[:i, :]


def convert_raw_event(events_raw_dir, new_dir, accumulate_time, finish_step):
    SENSOR_SIZE = (IMG_WIDTH, IMG_HEIGHT, 2)  # (WHP)
    finish_time = accumulate_time * finish_step + 1  # 最期の秒を記録 us?
    # converter = transforms.ToFrame(sensor_size=SENSOR_SIZE, time_window=accumulate_time)
    # converter = transforms.Compose(
    #     [transforms.ToFrame(sensor_size=SENSOR_SIZE, time_window=accumulate_time),
    #     num2torch(),
    #     Number2one(),
    #     transforms_.Resize(size=(INPUT_HEIGHT, INPUT_WIDTH),interpolation=T.InterpolationMode.NEAREST)]
    # )

    try:
        os.makedirs(new_dir)
        h5py_allfile = glob.glob(f"{events_raw_dir}/*.h5")
        h5py_leftfile = glob.glob(f"{events_raw_dir}/left/*.h5")
        h5py_rightfile = glob.glob(f"{events_raw_dir}/right/*.h5")
        dtype = [("t", "<i4"), ("x", "<i4"), ("y", "<i4"), ("p", "<i4")]
        # print(h5py_allfile)

        # # 0梅するための対策
        true_shape = (finish_step, 2, INPUT_HEIGHT, INPUT_WIDTH)

        if EVENT_COUNT:
            converter_event = transforms.Compose(
                [
                    transforms.ToFrame(
                        sensor_size=SENSOR_SIZE, time_window=accumulate_time
                    ),
                    num2torch(),
                    transforms_.Resize(
                        size=(INPUT_HEIGHT, INPUT_WIDTH),
                        interpolation=T.InterpolationMode.NEAREST,
                    ),
                    Fill0_Tensor(true_shape),
                ]
            )
        else:
            converter_event = transforms.Compose(
                [
                    transforms.ToFrame(
                        sensor_size=SENSOR_SIZE, time_window=accumulate_time
                    ),
                    num2torch(),
                    Number2one(),
                    transforms_.Resize(
                        size=(INPUT_HEIGHT, INPUT_WIDTH),
                        interpolation=T.InterpolationMode.NEAREST,
                    ),
                    Fill0_Tensor(true_shape),
                ]
            )

        # converter_label = transforms.Compose(
        #     [
        #         transforms_.ToTensor(),
        #         transforms_.Resize(
        #             size=(INPUT_HEIGHT, INPUT_WIDTH),
        #             interpolation=T.InterpolationMode.NEAREST,
        #         ),
        #         # ToRoughSegmentation(ROUGH_PIXEL),
        #     ]
        # )
        converter_label = transforms.Compose(
            [
                num2torch(),
                transforms_.Resize(
                    size=(INPUT_HEIGHT, INPUT_WIDTH),
                    interpolation=T.InterpolationMode.NEAREST,
                ),  # t, h, w
                # ToRoughSegmentation(ROUGH_PIXEL),
            ]
        )
        converter_rough_label = ToRoughSegmentation(ROUGH_PIXEL)
        for i, (leftfile, rightfile) in enumerate(
            zip(tqdm(h5py_leftfile), h5py_rightfile)
        ):
            with h5py.File(leftfile, "r") as f:
                label_left = f["label"][()]
                # label = f["label"][()]
                raw_events_left = f["events"][()]
            with h5py.File(rightfile, "r") as f:
                label_right = f["label"][()]
                raw_events_right = f["events"][()]

            # with h5py.File(file, "r") as f:
            #     label = f["label"][()]
            #     raw_events = f["events"][()]
            raw_events_left = get_until_finishtime(
                raw_events_left, finish_time=finish_time
            )
            raw_events_right = get_until_finishtime(
                raw_events_right, finish_time=finish_time
            )
            raw_events_len_left = raw_events_left.shape[0]
            raw_events_len_right = raw_events_right.shape[0]

            processed_events_left = np.zeros(raw_events_len_left, dtype=dtype)
            processed_events_right = np.zeros(raw_events_len_right, dtype=dtype)
            for idx, (key, _) in enumerate(dtype):
                processed_events_left[key] = raw_events_left[:, idx]
                processed_events_right[key] = raw_events_right[:, idx]

            if processed_events_left.shape[0] == 0:
                acc_events_left = np.zeros(true_shape)
            else:
                acc_events_left = converter_event(processed_events_left)

            if processed_events_right.shape[0] == 0:
                acc_events_right = np.zeros(true_shape)
            else:
                acc_events_right = converter_event(processed_events_right)
            # print(acc_events_left.shape, acc_events_right.shape)  # torch.Size([8, 2, 130, 173]) torch.Size([8, 2, 130, 173])
            # 鳥瞰図(真ん中のカメラから見た風景)からはみ出ている部分を削除。数字の決定はaffin.pyで手作業で行った。(実際はホモグラフィー変換とかやろうとしたけどうまくできず、、)
            # plt.subplot(2, 3, 1)
            # plt.imshow(acc_events_left[0, 0])
            # plt.subplot(2, 3, 2)
            # plt.imshow(acc_events_right[0, 0])
            # plt.show()
            # print(acc_events_left.shape, acc_events_right.shape)
            # right_idx = int(182 * INPUT_WIDTH // IMG_WIDTH)
            # left_idx = int(160 * INPUT_WIDTH // IMG_WIDTH)
            width = acc_events_left.shape[3]
            right_idx = RIGHT_IDX
            left_idx = LEFT_IDX
            acc_events_left = acc_events_left[
                :, :, :, :left_idx
            ]  # time, channel, height, width
            acc_events_right = acc_events_right[:, :, :, right_idx:]
            acc_events = np.concatenate([acc_events_right, acc_events_left], axis=3)
            # plt.show()
            # print(acc_events.shape)  # torch.Size([8, 2, 130, 160])
            # label = converter_label(label)
            label_left = converter_label(label_left)
            label_right = converter_label(label_right)
            # print(label_left.shape, label_right.shape)  # time, height, width
            label_left = label_left[:, :, :left_idx]
            label_right = label_right[:, :, right_idx:]
            label = torch.concatenate([label_right, label_left], axis=2)
            rough_label = converter_rough_label(label)
            file_name = f"{str(i).zfill(5)}.h5"
            new_file_path = os.path.join(new_dir, file_name)
            with h5py.File(new_file_path, "w") as f:
                f.create_dataset("label", data=rough_label)
                f.create_dataset("events", data=acc_events)
                f.create_dataset("label_fine", data=label)

    except Exception as e:
        import shutil

        shutil.rmtree(new_dir)
        import traceback

        traceback.print_exc()
        exit()
    return


class LoadDataset(Dataset):
    def __init__(
        self,
        processed_event_dataset_path,
        raw_event_dir,
        accumulate_time: int,
        input_height,
        input_width,
        finish_step,
        train: bool,
        test_rate=0.2,
        download=False,
    ):
        """
        processed_event_dataset_path: 処理済みのイベントデータのパス
        raw_event_dir: イベントの生データ
        accumulate_time: イベントの集積時間
        input_height, input_width: 入力画像の高さと幅
        finish_step: 何ステップをSNNに入力するか
        test_rate: 全体のデータに対するテストデータの割合
        train: 学習データかテストデータか
        download: データを一基にメモリに読み取らせるか。もちろんFalseにした方が早い。ただメモリエラーが起きたときはTrueにして、一回一回読み込ませるようにする(__getitem__時に読み込む必要がなくなるので早くなる)
        """  #
        self.accumulate_time = accumulate_time
        self.input_height = input_height
        self.input_width = input_width
        self.download = download

        # イベントデータの前処理を行う。すでにディレクトリがあったら飛ばす
        if os.path.isdir(processed_event_dataset_path):
            pass
        else:
            convert_raw_event(
                events_raw_dir=raw_event_dir,
                new_dir=processed_event_dataset_path,
                accumulate_time=self.accumulate_time,
                finish_step=MAX_STEP,
            )

        # 全てのイベントデータのファイルパスを取得
        self.all_files = glob.glob(f"{processed_event_dataset_path}/*")
        self.divide = int((len(self.all_files) * test_rate))

        if train:
            self.file_lst = self.all_files[self.divide :]
        else:
            self.file_lst = self.all_files[: self.divide]
        # データを丸ごとリストに格納する場合
        if self.download == False:
            self.all_data = []
            print("dataset読み込み開始")
            for path in tqdm(self.file_lst):
                with h5py.File(path, "r") as f:
                    label = f["label"][()]
                    input = f["events"][()]
                # print(input.shape, label.shape, label)
                input = input[:finish_step]
                label = label[:finish_step]
                input = torch.from_numpy(input.astype(np.float32)).clone()
                label = torch.tensor(label, dtype=torch.float32)
                # if label == 1:
                #     label = torch.tensor([0,1], dtype=torch.float32)
                # else:
                #     label = torch.tensor([1,0], dtype=torch.float32)
                self.all_data.append((input, label))
            print("dataset読み込み終了")

    def __len__(self):
        return len(self.file_lst)

    def __getitem__(self, index):
        if self.download:
            with h5py.File(self.file_lst[index], "r") as f:
                label = f["label"][()]
                input = f["events"][()]

            input = torch.from_numpy(input.astype(np.float32)).clone()
            label = torch.tensor(label, dtype=torch.float32)
        else:
            input, label = self.all_data[index]
        return input, label


class AnnLoadDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        input_height,
        input_width,
        train: bool,
        test_rate=0.2,
        download=False,
    ):
        self.input_height = input_height
        self.input_width = input_width
        self.download = download

        self.all_files = glob.glob(f"{dataset_dir}/*")
        self.divide = int((len(self.all_files) * test_rate))

        if train:
            self.file_lst = self.all_files[self.divide :]
        else:
            self.file_lst = self.all_files[: self.divide]
        if self.download == False:
            self.all_data = []
            print("dataset読み込み開始")
            for path in tqdm(self.file_lst):
                with h5py.File(path, "r") as f:
                    label = f["label"][()]
                    input = f["events"][()]
                input = torch.from_numpy(input.astype(np.float32)).clone()
                label = torch.from_numpy(label.astype(np.float32)).clone()
                self.all_data.append((input, label))
            print("dataset読み込み終了")

    def __len__(self):
        return len(self.file_lst)

    def __getitem__(self, index):
        if self.download:
            with h5py.File(self.file_lst[index], "r") as f:
                label = f["label"][()]
                input = f["events"][()]
            input = torch.from_numpy(input.astype(np.float32)).clone()
            label = torch.from_numpy(label.astype(np.float32)).clone()
        else:
            input, label = self.all_data[index]
        return input, label


def custom_collate(batch):
    """
    batchをどの引数に持っていくかを決める関数。入力はbatchを2つ目。ラベルは1つ目に設定。
    """
    input_lst = []
    target_lst = []
    for input, target in batch:
        input_lst.append(input)
        target_lst.append(target)
    return torch.stack(input_lst, dim=1), torch.stack(target_lst, dim=1)


if __name__ == "__main__":
    print(os.getcwd())
    dataset_path = "dataset/"
    raw_event_dir = "data"
    # youtube_path = "gomibako/h5.gif"
    a = LoadDataset(
        dir=dataset_path,
        raw_event_dir=raw_event_dir,
        accumulate_time=100000,
        train=True,
    )
    while 1:
        n = int(input())

        input, label = a[n]
        print(input.shape)
        print(label.shape)
        print(input.shape[0])
        # print(a.file_lst[6], a.num_lst[6])
    # make_data.youtube(input, youtube_path)
