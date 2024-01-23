import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

n = 12
csv_path = f"result_experiment/experiment_{n:03}.csv"
data = pd.read_csv(csv_path)

th = data.loc[:, "THRESHOLD"].unique()
weight = data.loc[:, "weight mean"].unique()
# 絶対値
weight = np.abs(weight)

plt.plot(th, weight, marker="o")
plt.show()
