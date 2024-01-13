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
    default=43,
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

data = data.rename(columns={"FINISH_STEP": "time step"})
data = data.rename(columns={"THRESHOLD": "Threshold"})

# 発火率の関係 when time step = 6
for i in range(4):
    data = data.rename(columns={f"spike_rate_{i}": f"layer{i+1}"})
print(data.head())

data = data[["layer1", "layer2", "layer3", "layer4", "Threshold"]]
print(data.head())
spike_rate_pivot_table = pd.melt(
    data,
    id_vars=["Threshold"],
    value_vars=["layer1", "layer2", "layer3", "layer4"],
    var_name="Layer",
    value_name="spike rate",
)
print(spike_rate_pivot_table)


# plt.figure(figsize=(15, 10))
g = sns.catplot(
    x="Layer",
    y="spike rate",
    hue="Threshold",
    data=spike_rate_pivot_table,
    kind="bar",  # 黒い線は95%信頼区間を示します
    palette="muted",
)
# figsize = (10, 10)
g.fig.set_size_inches(6, 4)
plt.savefig(os.path.join(SAVE_DIR, f"thresh_spike_rate.png"))
plt.savefig(os.path.join(SAVE_DIR, f"thresh_spike_rate.pdf"))
plt.show()
