# %%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

# %%

save_dir = "result_thesis/base"
os.makedirs(save_dir, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv_num",
    "-n",
    type=int,
    default=43,
)
args = parser.parse_args()
csv_num = args.csv_num
csv_path = f"result_experiment/experiment_{csv_num:03}.csv"
# csv_path = "result_experiment/experiment_031.csv"
data = pd.read_csv(csv_path)

# finish_step_lst = [2, 4, 6, 8]
finish_step_lst = [1, 2, 3, 4, 6]
threshhold = 1
leaky = 1
iou_lst = []
recall_lst = []
precision_lst = []
f_measure_lst = []

for i, timestep in enumerate(finish_step_lst):
    iou_lst.append(
        data.loc[
            (data["FINISH_STEP"] == timestep)
            & (data["THRESHOLD"] == threshhold)
            & (data["BETA"] == leaky),
            "IoU",
        ].values[0]
    )
    recall_lst.append(
        data.loc[
            (data["FINISH_STEP"] == timestep)
            & (data["THRESHOLD"] == threshhold)
            & (data["BETA"] == leaky),
            "Recall",
        ].values[0]
    )
    precision_lst.append(
        data.loc[
            (data["FINISH_STEP"] == timestep)
            & (data["THRESHOLD"] == threshhold)
            & (data["BETA"] == leaky),
            "Precision",
        ].values[0]
    )
    f_measure_lst.append(
        data.loc[
            (data["FINISH_STEP"] == timestep)
            & (data["THRESHOLD"] == threshhold)
            & (data["BETA"] == leaky),
            "F-Measure",
        ].values[0]
    )
plt.figure()
plt.plot(finish_step_lst, iou_lst, label="IoU", marker="x")
plt.plot(finish_step_lst, recall_lst, label="Recall", marker="x")
plt.plot(finish_step_lst, precision_lst, label="Precision", marker="x")
plt.plot(finish_step_lst, f_measure_lst, label="F-measure", marker="x")
plt.ylim(80, 100)
plt.legend()
plt.xlabel("time step")
plt.ylabel("Accuracy [%]")
plt.savefig(os.path.join(save_dir, "base_timestep.png"))
plt.savefig(os.path.join(save_dir, "base_timestep.pdf"))
plt.show()
