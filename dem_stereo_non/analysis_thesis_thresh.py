# %%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import seaborn as sns

# %%

parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv_num",
    "-n",
    type=int,
    default=6,
)
args = parser.parse_args()
csv_num = args.csv_num
csv_path = f"result_experiment/experiment_{csv_num:03}.csv"
# csv_path = "result_experiment/experiment_031.csv"
data = pd.read_csv(csv_path)
data = data.sort_values(by=["THRESHOLD", "BETA", "FINISH_STEP"])
# finish_step_lst = [2, 4, 6, 8]
threshhold_lst = data.loc[:, "THRESHOLD"].unique()
timestep_lst = data.loc[:, "FINISH_STEP"].unique()

SAVE_DIR = "result_thesis"
os.makedirs(SAVE_DIR, exist_ok=True)
# 0.25, 5の部分は除く
data = data[data["THRESHOLD"] != 0.25]
data = data[data["THRESHOLD"] != 5]
data = data.rename(columns={"FINISH_STEP": "time step"})
data = data.rename(columns={"THRESHOLD": "Threshold"})
iou_pivot_table = pd.pivot_table(
    index="Threshold",
    columns="time step",
    values="IoU",
    data=data,
    aggfunc="mean",
)
iou_pivot_table = iou_pivot_table.iloc[::-1]
sns.heatmap(
    iou_pivot_table,
    cmap="YlGn",
    annot=True,
    fmt=".1f",
    cbar_kws={"label": "IoU"},
    vmin=0,
    vmax=100,
)
plt.savefig(os.path.join(SAVE_DIR, f"none_thresh_time_iou.png"))
plt.savefig(os.path.join(SAVE_DIR, f"none_thresh_time_iou.pdf"))
plt.show()
plt.close()

precision_pivot_table = pd.pivot_table(
    index="Threshold",
    columns="time step",
    values="Precision",
    data=data,
    aggfunc="mean",
)
precision_pivot_table = precision_pivot_table.iloc[::-1]
sns.heatmap(
    precision_pivot_table,
    cmap="YlGn",
    annot=True,
    fmt=".1f",
    cbar_kws={"label": "Precision"},
    vmin=0,
    vmax=100,
)
plt.savefig(os.path.join(SAVE_DIR, f"none_thresh_time_precision.png"))
plt.savefig(os.path.join(SAVE_DIR, f"none_thresh_time_precision.pdf"))
plt.show()
plt.close()

recall_pivot_table = pd.pivot_table(
    index="Threshold",
    columns="time step",
    values="Recall",
    data=data,
    aggfunc="mean",
)
recall_pivot_table = recall_pivot_table.iloc[::-1]
sns.heatmap(
    recall_pivot_table,
    cmap="YlGn",
    annot=True,
    fmt=".1f",
    cbar_kws={"label": "Recall"},
    vmin=0,
    vmax=100,
)
plt.savefig(os.path.join(SAVE_DIR, f"none_thresh_time_recall.png"))
plt.savefig(os.path.join(SAVE_DIR, f"none_thresh_time_recall.pdf"))
plt.show()
plt.close()

fscore_pivot_table = pd.pivot_table(
    index="Threshold",
    columns="time step",
    values="F-Measure",
    data=data,
    aggfunc="mean",
)
fscore_pivot_table = fscore_pivot_table.iloc[::-1]
sns.heatmap(
    fscore_pivot_table,
    cmap="YlGn",
    annot=True,
    fmt=".1f",
    cbar_kws={"label": "F-measure"},
    vmin=0,
    vmax=100,
)
plt.savefig(os.path.join(SAVE_DIR, f"none_thresh_time_fmeasure.png"))
plt.savefig(os.path.join(SAVE_DIR, f"none_thresh_time_fmeasure.pdf"))
plt.show()
plt.close()

failed_area_pivot_table = pd.pivot_table(
    index="Threshold",
    columns="time step",
    values="Failed MaxArea",
    data=data,
    aggfunc="mean",
)
failed_area_pivot_table = failed_area_pivot_table.iloc[::-1]
# 小数点だから%表示させる
failed_area_pivot_table = failed_area_pivot_table * 100
# print(failed_area_pivot_table)
sns.heatmap(
    failed_area_pivot_table,
    cmap="YlGn",
    annot=True,
    fmt=".1f",
    cbar_kws={"label": "Failed MaxArea"},
    vmin=0,
    vmax=100,
)
plt.savefig(os.path.join(SAVE_DIR, f"none_leaky_time_failed_area.png"))
plt.savefig(os.path.join(SAVE_DIR, f"none_leaky_time_failed_area.pdf"))
plt.show()
plt.close()
# # %%
# # 発火率の関係 when time step = 6
# for i in range(4):
#     data = data.rename(columns={f"spike_rate_{i}": f"layer{i+1}"})
# fire_rate_pivot_table = pd.melt(
#     data,
#     id_vars=["layer1", "layer2", "layer3", "layer4"],
#     var_name="layers",
#     value_name="spike rate",
# )
# fire_rate_pivot_table.head()
