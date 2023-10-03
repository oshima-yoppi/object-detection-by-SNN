import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import utils
from snntorch import functional as SF
from snntorch import surrogate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tonic import DiskCachedDataset
import tonic

import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm

from module.custom_data import LoadDataset
from module import custom_data, compute_loss, network
from module.const import *
import matplotlib.pyplot as plt
from IPython.display import HTML

from collections import defaultdict

# import yaml
import time


def main():
    train_dataset = LoadDataset(
        processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH,
        raw_event_dir=RAW_EVENT_DIR,
        accumulate_time=ACCUMULATE_EVENT_MICROTIME,
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH,
        train=True,
        finish_step=START_STEP,
    )
    test_dataset = LoadDataset(
        processed_event_dataset_path=PROCESSED_EVENT_DATASET_PATH,
        raw_event_dir=RAW_EVENT_DIR,
        accumulate_time=ACCUMULATE_EVENT_MICROTIME,
        input_height=INPUT_HEIGHT,
        input_width=INPUT_WIDTH,
        train=False,
        finish_step=START_STEP,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=custom_data.custom_collate,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=custom_data.custom_collate,
        shuffle=False,
    )

    net = NET

    events, _ = train_dataset[0]
    num_steps = events.shape[0]
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))

    class_ration = torch.tensor([977, 1427], dtype=torch.float32).to(DEVICE)
    weights = 1.0 / class_ration
    weights = weights / weights.sum()
    # loss_func = nn.BCELoss(weight=weights)
    # loss_func = compute_loss.DiceLoss()
    loss_func = compute_loss.WeightedF1Loss(beta=1.5)
    analyzer = compute_loss.Analyzer()
    # loss_func = nn.BCELoss()

    num_epochs = 100
    # num_epochs = 2
    num_iters = 50
    # pixel = 64
    correct_rate = 0.5
    loss_hist = []
    hist = defaultdict(list)

    if TIME_CHANGE:
        time_step_lst = np.linspace(START_STEP, FINISH_STEP, 3).astype(int)
    else:
        time_step_lst = [FINISH_STEP]
    print(time_step_lst)
    # training loop
    # return

    model_save_path = MODEL_PATH
    max_acc = -1
    try:
        for time_step in time_step_lst:
            max_recall = -1
            for epoch in tqdm(range(num_epochs)):
                for i, (data, label) in enumerate(iter(train_loader)):
                    loss_log = []
                    data = data.to(DEVICE)
                    label = label.to(DEVICE)
                    batch = len(data[0])
                    # print(data.shape)
                    # data = data.reshape(num_steps, batch, INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH)
                    # print(data.shape)
                    net.train()
                    pred_pro = net(data, time_step)  # batch, channel, pixel ,pixel
                    # print(pred_pro.shape)
                    # print(label.shape)
                    # plt.figure()
                    # plt.imshow(pred_pro[0, 1].detach().cpu().numpy())
                    # plt.show()
                    # print(pred_pro.shape)
                    # print(pred_pro.shape)
                    # loss_val = criterion(pred_pro, label)
                    # loss_val = loss_func(pred_pro, label)
                    # loss_val = compute_loss.loss_dice(pred_pro, label, correct_rate)
                    loss_val = loss_func(pred_pro, label)
                    # loss_val = 1 - acc

                    # Gradient calculation + weight update
                    optimizer.zero_grad()
                    loss_val.backward()
                    optimizer.step()

                    # Store loss history for future plotting
                    loss_log.append(loss_val.item())
                    # hist["loss"].append(loss_val.item())
                hist["loss"].append(np.mean(loss_log))

                with torch.no_grad():
                    net.eval()
                    iou_log = []
                    precision_log = []
                    recall_log = []
                    for i, (data, label) in enumerate(iter(test_loader)):
                        data = data.to(DEVICE)
                        label = label.to(DEVICE)
                        batch = len(data[0])
                        # data = data.reshape(num_steps, batch, INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH)
                        pred_pro = net(data, time_step)

                        # pred_class = pred_pro.argmax(dim=1)
                        # label_class = label.argmax(dim=1)
                        # acc = (pred_class == label_class).sum().item() / batch
                        iou, precision, recall = analyzer(pred_pro, label)
                        iou_log.append(iou)
                        precision_log.append(precision)
                        recall_log.append(recall)
                    hist["iou"].append(np.mean(iou_log))
                    hist["precision"].append(np.mean(precision_log))
                    hist["recall"].append(np.mean(recall_log))
                    tqdm.write(
                        f"{epoch}:::  loss:{np.mean(loss_log)}, precision:{np.mean(precision_log)}, recall:{np.mean(recall_log)}"
                    )
                    # if max_recall < hist['recall'][-1] and hist['recall'][-1] > 0.5:
                    #     max_recall = hist['recall'][-1]
                    #     torch.save(net.state_dict(), model_save_path)
    except Exception as e:
        import traceback

        print("--------error--------")
        traceback.print_exc()
        print("--------error--------")
        pass
        # print(e)
    ## save model
    # enddir = MODEL_PATH
    # # if os.path.exists(enddir) == False:
    # #     os.makedirs(enddir)

    torch.save(net.state_dict(), model_save_path)
    print("success model saving")

    # print(MODEL_NAME)
    # print(f"{acc=}")
    # # Plot Loss
    # print(hist)
    # fig = plt.figure(facecolor="w")
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax2 = fig.add_subplot(1, 3, 2)
    # ax3 = fig.add_subplot(1, 3, 3)
    # ax1.plot(hist["loss"], label="train")
    # ax1.set_title("loss")
    # ax1.set_xlabel("Iteration")
    # ax1.set_ylabel("Loss (Dice)")
    # ax2.plot(hist["train"], label="train")
    # ax2.set_title("Train  accuracy")
    # ax2.set_xlabel("Iteration")
    # ax2.set_ylabel("Accuracy(IoU)")
    # ax3.plot(hist["test"], label="test")
    # ax3.set_title("Test acc")
    # ax3.set_xlabel("epoch")
    # ax3.set_ylabel("Accuracy(IoU)")
    # fig.suptitle(f"ModelName:{MODEL_NAME}")
    # fig.tight_layout()
    # plt.show()
    return hist


if __name__ == "__main__":
    main()
