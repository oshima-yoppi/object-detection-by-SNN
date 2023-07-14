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
import pandas as pd
from tqdm import tqdm

# from collections import defaultdict

from module.custom_data import LoadDataset
from module import custom_data, network, compute_loss, view
from module.const import *

import matplotlib.pyplot as plt
from IPython.display import HTML
from collections import defaultdict

import time


def main(hist=None,):
    # train_dataset = LoadDataset(processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH, raw_event_dir=RAW_EVENT_PATH, accumulate_time=ACCUMULATE_EVENT_MICROTIME , input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH,train=True, finish_step=FINISH_STEP)
    test_dataset = LoadDataset(
        processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH,
        raw_event_dir=RAW_EVENT_PATH,
        accumulate_time=ACCUMULATE_EVENT_MICROTIME,
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH,
        train=False,
        finish_step=FINISH_STEP,
    )

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TEST, collate_fn=custom_data.custom_collate, shuffle=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_TEST,
        collate_fn=custom_data.custom_collate,
        shuffle=False,
    )

    net = NET
    net.load_state_dict(torch.load(MODEL_PATH))
    # corract_rate  = 0.5

    # ious = []

    results = defaultdict(list)
    # results['iou'] = []

    # def analysis_segmentation(pred, label):
    #     # print(pred.shape)
    #     # print(label.shape)
    #     # exit()
    #     pred = pred.reshape((INPUT_HEIGHT, INPUT_WIDTH)).to('cpu').detach().numpy().copy()
    #     label = label.reshape((INPUT_HEIGHT, INPUT_WIDTH)).to('cpu').detach().numpy().copy()
    #     # print(pred.shape)
    #     # print(label.shape)
    #     # exit()
    #     all_pixel = INPUT_HEIGHT * INPUT_WIDTH
    #     TP = np.sum(np.where((pred>=CORRECT_RATE) & (label==1), 1, 0))/all_pixel
    #     TN = np.sum(np.where((pred<CORRECT_RATE) & (label==0), 1, 0))/all_pixel
    #     FP = np.sum(np.where((pred>=CORRECT_RATE) & (label==0), 1, 0))/all_pixel
    #     FN = np.sum(np.where((pred<CORRECT_RATE) & (label==1), 1, 0))/all_pixel
    #     # iou = compute_loss.culc_iou(pred, label, CORRECT_RATE)
    #     return TP, TN, FP, FN
    def save_train_process(path, hist):
        fig = plt.figure(facecolor="w")
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(hist["loss"], label="train")
        ax1.set_title("loss")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss (Dice)")
        ax2.plot(hist["acc"], label="acc")
        ax2.plot(hist["precision"], label="precision")
        ax2.plot(hist["recall"], label="recall")
        ax2.set_title("Test acc")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("Accuracy(IoU)")
        ax2.legend()
        fig.suptitle(f"ModelName:{MODEL_NAME}")
        fig.tight_layout()
        plt.savefig(path)

    def save_img(
        number, events, pred_pro, label, results, result_recall_path, pdf_output
    ):
        # label = label.reshape((pixel, pixel)).to('cpu')
        # print(pred_pro.shape)
        # number_str = str(number).zfill(5)

        fig = plt.figure()
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)

        # dem_filename = f'dem_{str(number).npy}'
        # dem_path = os.path.join(DEM_NP_PATH, dem_filename)
        # dem = np.load(dem_path)
        # ax1.imshow(dem)
        # print(number)
        # video_file_number = number // 4
        line_color = (150, 150, 0)
        number = str(number).zfill(5)
        video_filename = f"{number}.avi"
        video_path = os.path.join(VIDEO_PATH, video_filename)
        first_frame = view.get_first_frame(video_path)
        first_frame = view.draw_edge_of_areas(first_frame, line_color=line_color)

        ax2.set_title("Camera_view")
        # print(first_frame.shape)
        ax2.imshow(first_frame)

        first_events = view.get_first_events(events)
        first_events = view.draw_edge_of_areas(first_events, line_color=line_color)
        ax3.set_title("EVS view")
        ax3.imshow(first_events)

        label = (
            label.reshape((ROUGH_PIXEL, ROUGH_PIXEL)).to("cpu").detach().numpy().copy()
        )
        ax4.set_title("label")
        ax4.imshow(label)

        pred_pro_ = (
            pred_pro[0]
            .reshape((ROUGH_PIXEL, ROUGH_PIXEL))
            .to("cpu")
            .detach()
            .numpy()
            .copy()
        )
        # pred_pro_max = np.max(pred_pro_)
        # pred_pro_max_rounded = np.round(pred_pro_max, 2)
        # pred_pro_min = np.min(pred_pro_)
        # pred_pro_min_rounded = round(pred_pro_min, 2)
        ax5.set_title(f"pred_pro")
        ax5.imshow(pred_pro_, vmin=0, vmax=1)

        pred = torch.where(pred_pro[0] > CORRECT_RATE, 1, 0)
        pred = (
            pred.reshape((ROUGH_PIXEL, ROUGH_PIXEL)).to("cpu").detach().numpy().copy()
        )
        ax6.set_title("pred")
        ax6.imshow(pred)

        fig.suptitle(f"No.{number} ModelName:{MODEL_NAME}")
        plt.tight_layout()
        # plt.show()
        # exit()
        img_path = os.path.join(RESULT_PATH, f"{str(number).zfill(5)}.png")
        fig.savefig(img_path)

        if results["Recall"][-1] <= 0.9999:
            img_path = os.path.join(result_recall_path, f"{str(number).zfill(5)}.png")
            fig.savefig(img_path)

        if pdf_output:
            img_path = os.path.join(RESULT_PATH, f"{str(number).zfill(5)}.pdf")
            fig.savefig(img_path)
        # plt.show()
        plt.close()

        return

    if os.path.exists(RESULT_PATH):
        shutil.rmtree(RESULT_PATH)
    os.makedirs(RESULT_PATH)
    result_recall_path = os.path.join(RESULT_PATH, "recall_failed")
    os.makedirs(result_recall_path)
    if hist is not None:
        train_process_dir = os.path.join(RESULT_PATH, "process")
        os.makedirs(train_process_dir)
        train_process_path = os.path.join(train_process_dir, "train_process.png")
        save_train_process(train_process_path, hist)

    spikes_lst = []
    analyzer = compute_loss.Analyzer()
    with torch.no_grad():
        net.eval()
        for i, (events, label) in enumerate(tqdm(iter(test_loader))):
            events = events.to(DEVICE)
            label = label.to(DEVICE)
            pred_pro = net(events, FINISH_STEP)
            iou, prec, recall = analyzer(pred_pro, label)
            # print(iou, prec, recall)
            results["IoU"].append(iou)
            results["Precision"].append(prec)
            results["Recall"].append(recall)
            # print(iou, prec, recall)
            spikes_lst.append(net.spike_count)
            # s = time.time()
            save_img(
                i,
                events,
                pred_pro,
                label,
                results,
                result_recall_path,
                pdf_output=False,
            )
            # print(time.time()-s)

            # if i == 10:
            #     break

            # break
    # precision recall を求める

    results["Precision"] = np.mean(results["Precision"]) * 100
    results["Recall"] = np.mean(results["Recall"]) * 100
    results["IoU"] = np.mean(results["IoU"]) * 100

    results["Precision"] = round(results["Precision"], 2)
    results["Recall"] = round(results["Recall"], 2)
    results["IoU"] = round(results["IoU"], 2)

    # print(results)
    all_num_data = len(test_loader.dataset)
    # for key in ['TP', 'TN', 'FP', 'FN']:
    #     results[key] = results[key]/all_num_data * 100
    #     results[key] = round(results[key], 2)

    print(MODEL_NAME, results)

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
        video_width = net.input_width
        video_height = net.input_height
        for models in network_lst:
            for layer in models.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    neurons += video_height * video_width * layer.out_channels
        return neurons

    n_nerons = count_neuron(net)

    spike_rate = n_spikes / n_nerons
    results["Spike Rate"] = spike_rate.item()
    # results['Spike Rate'] = round(results['Spike Rate'], 2)


if __name__ == "__main__":
    main()
