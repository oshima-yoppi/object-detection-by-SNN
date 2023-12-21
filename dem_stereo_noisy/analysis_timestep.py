# %%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

# %%

parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv_num",
    "-n",
    type=int,
    default=31,
)
args = parser.parse_args()
csv_num = args.csv_num
csv_path = f"result_experiment/experiment_{csv_num:03}.csv"
# csv_path = "result_experiment/experiment_031.csv"
data = pd.read_csv(csv_path)

# finish_step_lst = [2, 4, 6, 8]
finish_step_lst = data.loc[:, "FINISH_STEP"].unique()
# leaky_lst = data.loc[:, "BETA"].unique()
repeat_input_lst = data.loc[:, "REPEAT_INPUT"].unique()
leaky = 1
plt.figure()
for i, repeat_input in enumerate(repeat_input_lst):
    precision = []
    recall = []
    iou = []
    f_measure = []

    for timestep in finish_step_lst:
        precision.append(
            data.loc[
                (data["FINISH_STEP"] == timestep)
                & (data["BETA"] == leaky)
                & (data["REPEAT_INPUT"] == repeat_input),
                "Precision",
            ].values[0]
        )
        recall.append(
            data.loc[
                (data["FINISH_STEP"] == timestep)
                & (data["BETA"] == leaky)
                & (data["REPEAT_INPUT"] == repeat_input),
                "Recall",
            ].values[0]
        )
        iou.append(
            data.loc[
                (data["FINISH_STEP"] == timestep)
                & (data["BETA"] == leaky)
                & (data["REPEAT_INPUT"] == repeat_input),
                "IoU",
            ].values[0]
        )
        f_measure.append(
            data.loc[
                (data["FINISH_STEP"] == timestep)
                & (data["BETA"] == leaky)
                & (data["REPEAT_INPUT"] == repeat_input),
                "F-Measure",
            ].values[0]
        )
    plt.subplot(1, len(repeat_input_lst), i + 1)
    plt.plot(finish_step_lst, precision, label="Precision", marker="x")
    plt.plot(finish_step_lst, recall, label="Recall", marker="x")
    plt.plot(finish_step_lst, iou, label="IoU", marker="x")
    plt.plot(finish_step_lst, f_measure, label="F-Measure", marker="x")
    plt.ylim(0, 100)
    plt.title(f"Repeat input: {repeat_input}")
    plt.legend()
# plt.figure()
plt.show()
