import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as SF
from snntorch import surrogate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import itertools
import cv2
from tqdm import tqdm

# from collections import defaultdict

from module.custom_data import LoadDataset
from module import custom_data, network, compute_loss, view
from module.const import *

import matplotlib.pyplot as plt
from IPython.display import HTML

from collections import defaultdict


def main():
    train_dataset = LoadDataset(
        processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH,
        raw_event_dir=RAW_EVENT_PATH,
        accumulate_time=ACCUMULATE_EVENT_MICROTIME,
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH,
        train=True,
        finish_step=FINISH_STEP,
    )
    test_dataset = LoadDataset(
        processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH,
        raw_event_dir=RAW_EVENT_PATH,
        accumulate_time=ACCUMULATE_EVENT_MICROTIME,
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH,
        train=False,
        finish_step=FINISH_STEP,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_TEST,
        collate_fn=custom_data.custom_collate,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_TEST,
        collate_fn=custom_data.custom_collate,
        shuffle=False,
    )

    net = NET
    net.load_state_dict(torch.load(MODEL_PATH))
    # corract_rate  = 0.5
    events, _ = train_dataset[0]
    num_steps = events.shape[0]

    ious = []

    def save_img(number, events, pred_pro, label, iou, pdf_output):
        # label = label.reshape((pixel, pixel)).to('cpu')
        # print(pred_pro.shape)
        number_str = str(number).zfill(5)
        num_steps = len(pred_pro)
        nrows = 2
        ncols = 5
        fig = plt.figure()
        # ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)

        # dem_filename = f'dem_{str(number).npy}'
        # dem_path = os.path.join(DEM_NP_PATH, dem_filename)
        # dem = np.load(dem_path)
        # ax1.imshow(dem)

        video_filename = f"{number_str}.avi"
        video_path = os.path.join(VIDEO_PATH, video_filename)
        first_frame = view.get_first_frame(video_path)
        ax2.set_title("Camera_view")
        ax2.imshow(first_frame)

        first_events = view.get_first_events(events)
        ax3.set_title("EVS view")
        ax3.imshow(first_events)

        label_ = label.reshape((INPUT_HEIGHT, INPUT_WIDTH)).to("cpu")
        ax4.imshow(label_)
        ax4.set_title("label")

        pred_pro_ = pred_pro[:, 1, :, :]
        pred_pro_ = (
            pred_pro_.reshape((INPUT_HEIGHT, INPUT_WIDTH))
            .to("cpu")
            .detach()
            .numpy()
            .copy()
        )
        ax5.imshow(pred_pro_)
        ax5.set_title("Estimated Probability")

        ax6.imshow(np.where(pred_pro_ >= CORRECT_RATE, 1, 0))
        ax6.set_title("Safe or Dangerous")

        fig.suptitle(f"No.{number} IoU:{round(iou, 3)}  ModelName:{MODEL_NAME}")
        plt.tight_layout()
        # plt.show()
        # exit()
        img_path = os.path.join(RESULT_PATH, f"{str(i).zfill(5)}.png")
        fig.savefig(img_path)
        if pdf_output:
            img_path = os.path.join(RESULT_PATH, f"{str(i).zfill(5)}.pdf")
            fig.savefig(img_path)
        plt.close()

    hist = defaultdict(list)
    if os.path.exists(RESULT_PATH):
        shutil.rmtree(RESULT_PATH)
    os.makedirs(RESULT_PATH)
    spikes_lst = []
    with torch.no_grad():
        net.eval()
        for i, (events, label) in enumerate(tqdm(iter(test_loader))):
            events = events.to(DEVICE)
            label = label.to(DEVICE)
            batch = len(events[0])
            # print(events.shape)# TBCHW
            # events = events.reshape(num_steps, batch, INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH)
            pred_pro = net(events, FINISH_STEP)

            iou = compute_loss.culc_iou(pred_pro, label, CORRECT_RATE)
            ious.append(iou)
            # pred_pro = compute_loss.show_pred(pred_pro, correct_rate)
            spikes_lst.append(net.spike_count)

            save_img(i, events, pred_pro, label, iou, pdf_output=False)
            # break

    results = {}
    # iouの平均を求める
    iou_mean = sum(ious) / len(ious)
    results["IoU"] = iou_mean
    print(MODEL_NAME, iou_mean)

    # スパイク数の平均を求める
    n_spikes = sum(spikes_lst) / len(spikes_lst)
    # results['Number of Spikes'] = n_spikes
    print(f"{n_spikes=}")

    # 1推論あたりのエネルギーを求める
    jules_per_spike = 0.9e-12  # J
    # jules_per_spike = 0.45e-9 #J hide
    jule_per_estimate = n_spikes * jules_per_spike
    results["Energy per inference"] = jule_per_estimate.item()
    print(f"{jule_per_estimate=}")

    # スパイクレート発火率を求める
    def count_neuron(net):
        network_lst = net.network_lst
        neurons = 0
        width = net.input_width
        height = net.input_height
        for models in network_lst:
            for layer in models.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    neurons += height * width * layer.out_channels
        return neurons

    n_nerons = count_neuron(net)

    spike_rate = n_spikes / n_nerons
    results["Spike Rate"] = spike_rate.item()

    return results


if __name__ == "__main__":
    main()
