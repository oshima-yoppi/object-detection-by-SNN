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

def main():
    train_dataset = LoadDataset(processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH, raw_event_dir=RAW_EVENT_PATH, accumulate_time=ACCUMULATE_EVENT_MICROTIME , input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH,train=True, finish_step=FINISH_STEP)
    test_dataset = LoadDataset(processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH, raw_event_dir=RAW_EVENT_PATH, accumulate_time=ACCUMULATE_EVENT_MICROTIME , input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, train=False, finish_step=FINISH_STEP)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TEST, collate_fn=custom_data.custom_collate, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, collate_fn=custom_data.custom_collate, shuffle=False,)



    net = NET
    net.load_state_dict(torch.load(MODEL_PATH))
    # corract_rate  = 0.5
    events, _ = train_dataset[0]
    num_steps = events.shape[0]

    # ious = []
    
    results = {}
    results['TP'] = []
    results['TN'] = []
    results['FP'] = []
    results['FN'] = []
    results['precision'] = []
    results['recall'] = []
    results['iou'] = []



    def analysis_segmentation(pred, label):
        # print(pred.shape)
        # print(label.shape)
        # exit()
        # print(pred.shape)
        pred_danger = pred[:,1]
        # print(pred_danger.shape)
        pred_danger = pred_danger.reshape((INPUT_HEIGHT, INPUT_WIDTH)).to('cpu').detach().numpy().copy()
        label_np = label.reshape((INPUT_HEIGHT, INPUT_WIDTH)).to('cpu').detach().numpy().copy()
        # print(pred.shape)
        # print(label.shape)
        # exit()
        all_pixel = INPUT_HEIGHT * INPUT_WIDTH
        TP = np.sum(np.where((pred_danger>=CORRECT_RATE) & (label_np==1), 1, 0))
        TN = np.sum(np.where((pred_danger<CORRECT_RATE) & (label_np==0), 1, 0))
        FP = np.sum(np.where((pred_danger>=CORRECT_RATE) & (label_np==0), 1, 0))
        FN = np.sum(np.where((pred_danger<CORRECT_RATE) & (label_np==1), 1, 0))

        iou = compute_loss.culc_iou(pred, label, CORRECT_RATE)
        eps = 1e-7
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        results['TP'].append(TP)
        results['TN'].append(TN)
        results['FP'].append(FP)
        results['FN'].append(FN)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['iou'].append(iou)
        return


    def save_img(number, events, pred_pro, label, results, pdf_output):
        # label = label.reshape((pixel, pixel)).to('cpu')
        # print(pred_pro.shape)
        number_str = str(number).zfill(5)
        tp = results['TP'][-1]
        tn = results['TN'][-1]
        fp = results['FP'][-1]
        fn = results['FN'][-1]
        iou = results['iou'][-1]
        precision = results['precision'][-1]
        recall = results['recall'][-1]

        precision_ = np.round(precision, 3)
        recall_ = np.round(recall, 3)

        

        fig = plt.figure()
        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)


        # 1 表作成
        data  = np.array([[tp, tn], 
                          [fp, fn],
                          [precision_, recall_],
                          ])
        row_labels = ['True', 'False', 'Pre/Re']
        col_labels = ['Positive', 'Negative']
        ax1.axis('off')
        ax1.table(cellText=data, rowLabels=row_labels, colLabels=col_labels, loc='center')
        ax1.set_title(f'All pixels:{INPUT_HEIGHT*INPUT_WIDTH}')

        # 動画の最初のフレームを表示
        video_filename = f'{number_str}.avi'
        video_path = os.path.join(VIDEO_PATH, video_filename)
        first_frame = view.get_first_frame(video_path) 
        ax2.set_title('Camera_view')
        ax2.imshow(first_frame)

        # イベントカメラの最初のフレームを表示
        first_events = view.get_first_events(events) 
        ax3.set_title('EVS view')
        ax3.imshow(first_events)

        # ラベルを表示
        label_ =label.reshape((INPUT_HEIGHT, INPUT_WIDTH)).to('cpu')
        ax4.imshow(label_)
        ax4.set_title('label')

        # 危険ドの確率を表示
        pred_pro_ = pred_pro[:, 1, :, :]
        pred_pro_ = pred_pro_.reshape((INPUT_HEIGHT, INPUT_WIDTH)).to('cpu').detach().numpy().copy()
        ax5.imshow(pred_pro_)
        ax5.set_title('Estimated Probability')

        # 危険かどうかを表示
        ax6.imshow(np.where(pred_pro_>=CORRECT_RATE, 1, 0))
        ax6.set_title('Safe or Dangerous')

        fig.suptitle(f"No.{number} IoU:{round(iou, 3)}  ModelName:{MODEL_NAME}")
        plt.tight_layout()
        # plt.show()
        # exit()
        img_path = os.path.join(RESULT_PATH, f'{str(i).zfill(5)}.png')
        fig.savefig(img_path)
        if pdf_output:
            img_path = os.path.join(RESULT_PATH, f'{str(i).zfill(5)}.pdf')
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
            analysis_segmentation(pred_pro, label)
            # iou = compute_loss.culc_iou(pred_pro, label, CORRECT_RATE)
            
            # pred_pro = compute_loss.show_pred(pred_pro, correct_rate)
            spikes_lst.append(net.spike_count)  
        
            save_img(i, events, pred_pro, label,  results, pdf_output=False)
            # break
    results_mean = defaultdict(list)
    # iouの平均を求める
    iou_mean = sum(results['iou'])/len(results['iou'])
    results_mean['IoU'] = iou_mean
    print(MODEL_NAME, iou_mean)

    # スパイク数の平均を求める
    n_spikes = sum(spikes_lst)/len(spikes_lst)
    # results['Number of Spikes'] = n_spikes
    print(f'{n_spikes=}')

    # 1推論あたりのエネルギーを求める
    jules_per_spike = 0.9e-12 #J
    # jules_per_spike = 0.45e-9 #J hide
    jule_per_estimate = n_spikes*jules_per_spike
    results_mean['Energy per inference'] = jule_per_estimate.item()
    print(f'{jule_per_estimate=}')

    # スパイクレート発火率を求める
    def count_neuron(net):
        network_lst = net.network_lst
        neurons = 0
        width = net.input_width
        height = net.input_height
        for models in network_lst:
            for layer in models.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    neurons += height* width * layer.out_channels
        return neurons
    n_nerons = count_neuron(net)

    spike_rate = n_spikes/n_nerons
    results_mean['Spike Rate'] = spike_rate.item()

    results_mean['TP'] = sum(results['TP'])/len(results['TP'])
    results_mean['TN'] = sum(results['TN'])/len(results['TN'])
    results_mean['FP'] = sum(results['FP'])/len(results['FP'])
    results_mean['FN'] = sum(results['FN'])/len(results['FN'])
    results_mean['Precision'] = sum(results['precision'])/len(results['precision'])
    results_mean['Recall'] = sum(results['recall'])/len(results['recall'])
    return results_mean



if __name__ == "__main__":
    main()