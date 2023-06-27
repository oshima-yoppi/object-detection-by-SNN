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
def main(classification=False):
    # train_dataset = LoadDataset(processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH, raw_event_dir=RAW_EVENT_PATH, accumulate_time=ACCUMULATE_EVENT_MICROTIME , input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH,train=True, finish_step=FINISH_STEP)
    test_dataset = LoadDataset(processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH, raw_event_dir=RAW_EVENT_PATH, accumulate_time=ACCUMULATE_EVENT_MICROTIME , input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, train=False, finish_step=FINISH_STEP)


    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TEST, collate_fn=custom_data.custom_collate, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, collate_fn=custom_data.custom_collate, shuffle=False,)



    net = NET
    net.load_state_dict(torch.load(MODEL_PATH))
    # corract_rate  = 0.5
   

    # ious = []
    
    # results = defaultdict(list)
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


    def save_img(number, events, pred_pro, label_class, bool_pred,pdf_output):
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
        video_file_number = number // 9
        video_file_number = str(video_file_number).zfill(5)
        video_filename = f'{video_file_number}.avi'
        video_path = os.path.join(VIDEO_PATH, video_filename)
        first_frame = view.get_first_frame(video_path) 
        # print(first_frame.shape)
        i, j = number % 9 // 3, number % 9 % 3
        video_height, video_width, _ = first_frame.shape
        first_frame = first_frame[i*video_height//3:(i+1)*video_height//3, j*video_width//3:(j+1)*video_width//3]
        # if number % 4 == 0:
        #     first_frame = first_frame[0:video_height//2, :video_width//2]
        # elif number % 4 == 1:
        #     first_frame = first_frame[0:video_height//2, video_width//2:]
        # elif number % 4 == 2:
        #     first_frame = first_frame[video_height//2:, :video_width//2]
        # elif number % 4 == 3:
        #     first_frame = first_frame[video_height//2:, video_width//2:]
        
        danger_pro = pred_pro[0, 1].item()
        danger_pro = round(danger_pro*100, 2)

        ax1.set_title('Camera_view')
        ax1.imshow(first_frame)

        first_events = view.get_first_events(events) 
        ax2.set_title('EVS view')
        ax2.imshow(first_events)

        fig.suptitle(f"No.{number} __ {bool_pred}_ label_class:{label_class.item()}  danger:{danger_pro}%")
        
        plt.tight_layout()
        # plt.show()
        # exit()
        img_path = os.path.join(RESULT_PATH, f'{str(number).zfill(5)}.png')
        fig.savefig(img_path)
        if bool_pred == 'FP':
            shutil.copy(img_path, result_FP_path)
        elif bool_pred == 'FN':
            shutil.copy(img_path, result_FN_path)
        if pdf_output:
            img_path = os.path.join(RESULT_PATH, f'{str(number).zfill(5)}.pdf')
            fig.savefig(img_path)
        # plt.show()
        plt.close()
        
        return
    

    if os.path.exists(RESULT_PATH):
        shutil.rmtree(RESULT_PATH)
    os.makedirs(RESULT_PATH)
    # result_recall_path = os.path.join(RESULT_PATH, 'recall_failed')
    # os.makedirs(result_recall_path)
    result_FN_path = os.path.join(RESULT_PATH, 'FN_images')
    result_FP_path = os.path.join(RESULT_PATH, 'FP_images')
    os.makedirs(result_FN_path)
    os.makedirs(result_FP_path)

    spikes_lst = []
    # analyzer = compute_loss.Analyzer()
    with torch.no_grad():
        net.eval()
        for i, (events, label) in enumerate(tqdm(iter(test_loader))):
            events = events.to(DEVICE)
            label = label.to(DEVICE)
            # batch = len(events[0])
            # print(events.shape)# TBCHW
            # events = events.reshape(num_steps, batch, INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH)
            pred_pro = net(events, FINISH_STEP)
            pred_class = pred_pro.argmax(dim=1)
            label_class = label.argmax(dim=1)
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
            # s = time.time()
            save_img(i, events, pred_pro, label_class, bool_pred,  pdf_output=False)
            # print(time.time()-s)

            # if i == 10:
            #     break

            # break
    # precision recall を求める
    eps  = 1e-7
    # results['Precision'] = results['TP']/(results['TP']+results['FP'] + eps) * 100
    # results['Recall'] = results['TP']/(results['TP']+results['FN'] + eps) * 100
    # results['Accuracy'] = (results['TP']+results['TN'])/(results['TP']+results['TN']+results['FP']+results['FN'] + eps) * 100
    results['Precision'] = np.mean(results['Precision']) * 100
    results['Recall'] = np.mean(results['Recall']) * 100
    results['IoU'] = np.mean(results['IoU']) * 100

    results['Precision'] = round(results['Precision'], 2)
    results['Recall'] = round(results['Recall'], 2)
    results['IoU'] = round(results['IoU'], 2)


    # print(results)
    all_num_data = len(test_loader.dataset)
    # for key in ['TP', 'TN', 'FP', 'FN']:
    #     results[key] = results[key]/all_num_data * 100
    #     results[key] = round(results[key], 2)

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
    # results['Spike Rate'] = round(results['Spike Rate'], 2)
    if classification == False:
        return results
    



if __name__ == '__main__':
    main()