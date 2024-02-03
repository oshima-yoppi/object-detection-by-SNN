# %%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

# %%

save_dir = "result_thesis/base"
os.makedirs(save_dir, exist_ok=True)

csv_path_lst = [
    "../dem_stereo_non/result_experiment/experiment_012.csv",
    "../dem_stereo/result_experiment/experiment_057.csv",
    "../dem_stereo_noisy/result_experiment/experiment_021.csv",
]
# plt.figure()
iou_lst = []
recall_lst = []
precision_lst = []
f_measure_lst = []
failed_max_area_rate_lst = []

for csv_path in csv_path_lst:
    print(csv_path)
    # csv_path = "result_experiment/experiment_031.csv"
    data = pd.read_csv(csv_path)

    # finish_step_lst = [2, 4, 6, 8]
    finish_step_lst = [1, 2, 3, 4, 6]
    threshhold = 1
    leaky = 1
    iou = []
    recall = []
    precision = []
    f_measure = []
    failed_max_area_rate = []
    data["Failed MaxArea"] = data["Failed MaxArea"] * 100
    for i, timestep in enumerate(finish_step_lst):
        iou.append(
            data.loc[
                (data["FINISH_STEP"] == timestep)
                & (data["THRESHOLD"] == threshhold)
                & (data["BETA"] == leaky),
                "IoU",
            ].values[0]
        )
        recall.append(
            data.loc[
                (data["FINISH_STEP"] == timestep)
                & (data["THRESHOLD"] == threshhold)
                & (data["BETA"] == leaky),
                "Recall",
            ].values[0]
        )
        precision.append(
            data.loc[
                (data["FINISH_STEP"] == timestep)
                & (data["THRESHOLD"] == threshhold)
                & (data["BETA"] == leaky),
                "Precision",
            ].values[0]
        )
        f_measure.append(
            data.loc[
                (data["FINISH_STEP"] == timestep)
                & (data["THRESHOLD"] == threshhold)
                & (data["BETA"] == leaky),
                "F-Measure",
            ].values[0]
        )
        failed_max_area_rate.append(
            data.loc[
                (data["FINISH_STEP"] == timestep)
                & (data["THRESHOLD"] == threshhold)
                & (data["BETA"] == leaky),
                "Failed MaxArea",
            ].values[0]
        )
    # plt.figure()
    iou_lst.append(iou)
    recall_lst.append(recall)
    precision_lst.append(precision)
    f_measure_lst.append(f_measure)
    failed_max_area_rate_lst.append(failed_max_area_rate)

plt.figure(figsize=(13, 5))
plt.subplot(1, 4, 1)
plt.plot(finish_step_lst, iou_lst[0], label="Small Noise", marker="x")
plt.plot(finish_step_lst, iou_lst[1], label="Midium  Noise", marker="x")
plt.plot(finish_step_lst, iou_lst[2], label="Large Noise", marker="x")
plt.xlabel("time step")
plt.ylabel("IoU [%]")
plt.ylim(50, 100)
plt.legend(loc="lower right")
plt.subplot(1, 4, 2)
plt.plot(finish_step_lst, recall_lst[0], label="Small Noise", marker="x")
plt.plot(finish_step_lst, recall_lst[1], label="Midium  Noise", marker="x")
plt.plot(finish_step_lst, recall_lst[2], label="Large Noise", marker="x")
plt.xlabel("time step")
plt.ylabel("Recall [%]")
plt.ylim(50, 100)
plt.legend(loc="lower right")
plt.subplot(1, 4, 3)
plt.plot(finish_step_lst, precision_lst[0], label="Small Noise", marker="x")

plt.plot(finish_step_lst, precision_lst[1], label="Midium  Noise", marker="x")
plt.plot(finish_step_lst, precision_lst[2], label="Large Noise", marker="x")
plt.xlabel("time step")
plt.ylabel("Precision [%]")
plt.ylim(50, 100)
plt.legend(loc="lower right")
plt.subplot(1, 4, 4)

plt.plot(finish_step_lst, f_measure_lst[0], label="Small Noise", marker="x")
plt.plot(finish_step_lst, f_measure_lst[1], label="Midium  Noise", marker="x")
plt.plot(finish_step_lst, f_measure_lst[2], label="Large Noise", marker="x")
plt.xlabel("time step")
plt.ylabel("F-measure [%]")
plt.ylim(50, 100)
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "base_noise.png"))
plt.savefig(os.path.join(save_dir, "base_noise.pdf"))
plt.show()

plt.plot(finish_step_lst, failed_max_area_rate_lst[0], label="Small Noise", marker="x")
plt.plot(
    finish_step_lst, failed_max_area_rate_lst[1], label="Midium  Noise", marker="x"
)
plt.plot(finish_step_lst, failed_max_area_rate_lst[2], label="Large Noise", marker="x")
plt.xlabel("time step")
plt.ylabel("Failed MaxArea [%]")
plt.ylim(0, 50)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "base_noise_failed_max_area.png"))
plt.savefig(os.path.join(save_dir, "base_noise_failed_max_area.pdf"))
plt.show()
