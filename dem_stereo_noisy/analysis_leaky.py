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
data = pd.read_csv(csv_path)


# beta_lst = [0.2, 0.4, 0.6, 0.8, 1.0]
beta_lst = data["BETA"].unique()
beta_lst.sort()
# finish_step_lst = [1, 2, 4, 5, 6, 8]

# finish_step_lst = [1, 2, 4, 5, 6, 8]
finish_step = 3

precision = []
recall = []
iou = []
f_measure = []
for beta in beta_lst:
    precision.append(
        data.loc[
            (data["FINISH_STEP"] == finish_step) & (data["BETA"] == beta),
            "Precision",
        ].values[0]
    )
    recall.append(
        data.loc[
            (data["FINISH_STEP"] == finish_step) & (data["BETA"] == beta),
            "Recall",
        ].values[0]
    )
    iou.append(
        data.loc[
            (data["FINISH_STEP"] == finish_step) & (data["BETA"] == beta),
            "IoU",
        ].values[0]
    )
    f_measure.append(
        data.loc[
            (data["FINISH_STEP"] == finish_step) & (data["BETA"] == beta),
            "F-Measure",
        ].values[0]
    )

plt.plot(beta_lst, precision, label="Precision", marker="x")
plt.plot(beta_lst, recall, label="Recall", marker="x")
plt.plot(beta_lst, iou, label="IoU", marker="x")
plt.plot(beta_lst, f_measure, label="F-Measure", marker="x")
plt.xlim(0, 1)
plt.ylim(70, 100)
plt.legend()
plt.xlabel("Leaky")
plt.ylabel("Accuracu[%]")
result_saev_dir = "thesis_result"
import os

os.makedirs(result_saev_dir, exist_ok=True)
plt.savefig(f"{result_saev_dir}/leaky.png")
plt.savefig(f"{result_saev_dir}/leaky.pdf")

plt.show()
