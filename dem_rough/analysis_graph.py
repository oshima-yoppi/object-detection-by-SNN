# %%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

# %%
csv_path = "result_experiment/experiment_011.csv"
data = pd.read_csv(csv_path)


# accuracy = []
event_th = [0.05, 0.1, 0.15]
time_change_lst = [False, True]
for i, time_change in enumerate(time_change_lst):
    for j, th in enumerate(data["EVENT_TH"].unique()):
        precision = []
        recall = []
        for fin_step in data["FINISH_STEP"].unique():
            precision.append(
                data.loc[(data["EVENT_TH"] == th) & (data["FINISH_STEP"] == fin_step) & (data["TIME_CHANGE"] == time_change), "Precision"].values[0]
            )
            recall.append(
                data.loc[(data["EVENT_TH"] == th) & (data["FINISH_STEP"] == fin_step) & (data["TIME_CHANGE"] == time_change), "Recall"].values[0]
            )
        plt_width = len(data["EVENT_TH"].unique())
        plt.subplot(len(time_change_lst), plt_width, (i) * plt_width + j + 1)
        plt.plot(data["FINISH_STEP"].unique(), precision, label="Precision", linestyle="solid", marker="x", color="red")
        plt.plot(data["FINISH_STEP"].unique(), recall, label="Recall", linestyle="solid", marker="x", color="blue")
        plt.ylim(50, 100)
        plt.xlim(1, 8)
        plt.title("EVENT_TH: {}".format(th))


plt.legend()
plt.tight_layout()
plt.show()
# %%
