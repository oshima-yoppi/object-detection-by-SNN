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
threshhold_lst = data.loc[:, "THRESHOLD"].unique()
print(finish_step_lst, threshhold_lst)
leaky = 1
# plt.figure()
for i, thresh in enumerate(threshhold_lst):
    precision = []
    recall = []
    iou = []
    f_measure = []
    for step in finish_step_lst:
        print(8888)
        precision.append(
            data.loc[
                (data["FINISH_STEP"] == step)
                & (data["THRESHOLD"] == thresh)
                & (data["BETA"] == leaky),
                "Precision",
            ].values[0]
        )
        recall.append(
            data.loc[
                (data["FINISH_STEP"] == step)
                & (data["THRESHOLD"] == thresh)
                & (data["BETA"] == leaky),
                "Recall",
            ].values[0]
        )
        iou.append(
            data.loc[
                (data["FINISH_STEP"] == step)
                & (data["THRESHOLD"] == thresh)
                & (data["BETA"] == leaky),
                "IoU",
            ].values[0]
        )
        f_measure.append(
            data.loc[
                (data["FINISH_STEP"] == step)
                & (data["THRESHOLD"] == thresh)
                & (data["BETA"] == leaky),
                "F-Measure",
            ].values[0]
        )

    plt.figure()
    plt.plot(finish_step_lst, precision, label="Precision", marker="x")
    plt.plot(finish_step_lst, recall, label="Recall", marker="x")
    plt.plot(finish_step_lst, iou, label="IoU", marker="x")
    plt.plot(finish_step_lst, f_measure, label="F-Measure", marker="x")
    plt.legend()
    plt.ylim(75, 100)
    plt.title(f"threshhold: {thresh}")
    plt.xlabel("Timestep")
    plt.ylabel("Accuracy[%]")
    plt.show()
