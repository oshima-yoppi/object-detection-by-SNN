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


def main(classification=False):
    # train_dataset = LoadDataset(processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH, raw_event_dir=RAW_EVENT_PATH, accumulate_time=ACCUMULATE_EVENT_MICROTIME , input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH,train=True, finish_step=FINISH_STEP)
    test_dataset = LoadDataset(processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH, raw_event_dir=RAW_EVENT_PATH, accumulate_time=ACCUMULATE_EVENT_MICROTIME , input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, train=False, finish_step=FINISH_STEP)


    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TEST, collate_fn=custom_data.custom_collate, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, collate_fn=custom_data.custom_collate, shuffle=False,)



    net = NET
    net.load_state_dict(torch.load(MODEL_PATH))
    # corract_rate  = 0.5
   

    # ious = []
    
    results = defaultdict(int)
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


    def save_img(number, events, pred_pro, label, bool_pred, pdf_output):
        # label = label.reshape((pixel, pixel)).to('cpu')
        # print(pred_pro.shape)
        # number_str = str(number).zfill(5)
        

    
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)


        # dem_filename = f'dem_{str(number).npy}'
        # dem_path = os.path.join(DEM_NP_PATH, dem_filename)
        # dem = np.load(dem_path)
        # ax1.imshow(dem)
        # print(number)
        video_file_number = number // 4
        video_file_number = str(video_file_number).zfill(5)
        video_filename = f'{video_file_number}.avi'
        video_path = os.path.join(VIDEO_PATH, video_filename)
        first_frame = view.get_first_frame(video_path) 
        # print(first_frame.shape)
        video_height, video_width, _ = first_frame.shape
        if number % 4 == 0:
            first_frame = first_frame[0:video_height//2, :video_width//2]
        elif number % 4 == 1:
            first_frame = first_frame[0:video_height//2, video_width//2:]
        elif number % 4 == 2:
            first_frame = first_frame[video_height//2:, :video_width//2]
        elif number % 4 == 3:
            first_frame = first_frame[video_height//2:, video_width//2:]
        
        ax1.set_title('Camera_view')
        ax1.imshow(first_frame)

        first_events = view.get_first_events(events) 
        ax2.set_title('EVS view')
        ax2.imshow(first_events)

        fig.suptitle(f"No.{number} __ {bool_pred}")
        plt.tight_layout()
        # plt.show()
        # exit()
        img_path = os.path.join(RESULT_PATH, f'{str(i).zfill(5)}.png')
        fig.savefig(img_path)
        if pdf_output:
            img_path = os.path.join(RESULT_PATH, f'{str(i).zfill(5)}.pdf')
            fig.savefig(img_path)
        # plt.show()
        plt.close()
        
        return
    

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
            pred_class = torch.argmax(pred_pro, dim=1)
            label_class = torch.argmax(label, dim=1)
            # print(label_class, i)
            if pred_class == 1:
                if label_class == 1:
                    bool_pred = 'TP'
                else:
                    bool_pred = 'FP'
            else:
                if label_class == 1:
                    bool_pred = 'FN'
                else:
                    bool_pred = 'TN'
            results[bool_pred] += 1
            spikes_lst.append(net.spike_count)  
        
            save_img(i, events, pred_pro, label, bool_pred, pdf_output=False)

            # if i == 10:
            #     break

            # break

    # iouの平均を求める

    all_num_data = len(test_loader.dataset)
    for key in results.keys():
        results[key] = results[key]/all_num_data * 100
        results[key] = round(results[key], 2)

    print(MODEL_NAME, results)

    # スパイク数の平均を求める
    n_spikes = sum(spikes_lst)/len(spikes_lst)
    # results['Number of Spikes'] = n_spikes
    print(f'{n_spikes=}')

    # 1推論あたりのエネルギーを求める
    jules_per_spike = 0.9e-12 #J
    # jules_per_spike = 0.45e-9 #J hide
    jule_per_estimate = n_spikes*jules_per_spike
    results['Energy per inference'] = jule_per_estimate.item()
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
    results['Spike Rate'] = spike_rate.item()
    if classification == False:
        return results
    



if __name__ == '__main__':
    main()