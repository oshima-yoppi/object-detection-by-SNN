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
import h5py
import time


def main(
    hist=None,
    pdf_output=False,
):
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
    area_recall = dict()
    for area in range(100 * 100):
        area_recall[area] = [0, 0]  # failed num, all num

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
        ax3 = fig.add_subplot(234)
        ax4 = fig.add_subplot(233)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)

        line_color = (150, 150, 0)
        number = str(number).zfill(5)
        video_filename = f"{number}.avi"
        center_video_path = os.path.join(VIDEO_CENTER_PATH, video_filename)
        right_video_path = os.path.join(VIDEO_RIGHT_PATH, video_filename)
        left_video_path = os.path.join(VIDEO_LEFT_PATH, video_filename)
        center_first_frame = view.get_first_frame(center_video_path)

        center_first_frame = view.draw_edge_of_areas(
            center_first_frame, line_color=line_color
        )
        ax1.set_title("center_view")
        ax1.imshow(center_first_frame)
        right_first_frame = view.get_first_frame(right_video_path)
        left_first_frame = view.get_first_frame(left_video_path)
        right_first_frame = right_first_frame[:, RIGHT_IDX * 2 :, :]
        left_first_frame = left_first_frame[:, : LEFT_IDX * 2, :]
        # plt.subplot(121)
        # plt.imshow(right_first_frame)
        # plt.subplot(122)
        # plt.imshow(left_first_frame)
        # plt.show()
        concated_frame = np.concatenate((right_first_frame, left_first_frame), axis=1)
        concated_frame = view.draw_edge_of_areas(concated_frame, line_color=line_color)

        ax2.set_title("concated_view")
        ax2.imshow(concated_frame)

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
        ax6.imshow(pred, vmin=0, vmax=1)

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

    def get_area_boulder_lst(number):
        fine_label_path = os.path.join(
            PROCESSED_EVENT_DATASET_PATH, f"{str(number).zfill(5)}.h5"
        )
        with h5py.File(fine_label_path, "r") as f:
            fine_label = f["label_fine"][:]
        fine_label = np.squeeze(fine_label)
        # if number ==20:
        #     print(fine_label.shape)
        # print(fine_label.shape)
        areas_lst = []
        splited_width = fine_label.shape[1] // ROUGH_PIXEL
        splited_height = fine_label.shape[0] // ROUGH_PIXEL
        # print(fine_label.shape)
        for tate in range(ROUGH_PIXEL):
            for yoko in range(ROUGH_PIXEL):
                splited_area = fine_label[
                    splited_height * tate : splited_height * (tate + 1),
                    splited_width * yoko : splited_width * (yoko + 1),
                ]
                # print(splited_area.shape)
                area_of_boulder = np.sum(splited_area)
                areas_lst.append(area_of_boulder)

        return areas_lst

    if os.path.exists(RESULT_PATH):
        shutil.rmtree(RESULT_PATH)
    os.makedirs(RESULT_PATH)
    result_recall_path = os.path.join(RESULT_PATH, "recall_failed")
    result_area_path = os.path.join(RESULT_PATH, "area")
    os.makedirs(result_recall_path)
    os.makedirs(result_area_path)
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

            binary_result = analyzer.pred_binary
            target = analyzer.target
            areas_of_boulder_lst = get_area_boulder_lst(i)
            for area, p, t in zip(areas_of_boulder_lst, binary_result, target):
                if area == 242:
                    print(i)
                if t == 1:
                    area_recall[area][1] += 1
                    if p == 1:
                        area_recall[area][0] += 1
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
                pdf_output=pdf_output,
            )

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

    n_neurons = net.count_neurons()

    spike_rate = n_spikes / n_neurons
    results["Spike Rate"] = spike_rate.item()
    # results['Spike Rate'] = round(results['Spike Rate'], 2)
    # print(area_recall)
    plt.figure()
    for key in area_recall.keys():
        if area_recall[key][1] != 0:
            if area_recall[key][0] / area_recall[key][1] < 0.9999:
                print(
                    key,
                    area_recall[key][0] / area_recall[key][1],
                    area_recall[key][0],
                    area_recall[key][1],
                )
            plt.plot(key / 43 / 54, area_recall[key][0] / area_recall[key][1], "o")
        else:
            # plt.plot(key / 43 / 54, 1, "o")
            pass

    # plt.show()
    plt.savefig(os.path.join(result_area_path, "area_recall.png"))
    plt.savefig(os.path.join(result_area_path, "area_recall.pdf"))
    plt.close()
    return results


if __name__ == "__main__":
    main(pdf_output=False)
