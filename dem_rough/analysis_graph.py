# %%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

# %%
csv_path = "result_experiment\experiment_009.csv"
data = pd.read_csv(csv_path)


# accuracy = []
event_th = [0.05, 0.1, 0.15]
for i, th in enumerate(data["EVENT_TH"].unique()):
    precision = []
    recall = []
    for fin_step in data["FINISH_STEP"].unique():
        precision.append(data.loc[(data["EVENT_TH"] == th) & (data["FINISH_STEP"] == fin_step), "Precision"].values[0])
        recall.append(data.loc[(data["EVENT_TH"] == th) & (data["FINISH_STEP"] == fin_step), "Recall"].values[0])
    plt.subplot(1, len(data["EVENT_TH"].unique()), i + 1)
    plt.plot(data["FINISH_STEP"].unique(), precision, label="Precision", linestyle="solid", marker="x", color="red")
    plt.plot(data["FINISH_STEP"].unique(), recall, label="Recall", linestyle="solid", marker="x", color="blue")
    plt.ylim(80, 100)
    plt.title("EVENT_TH: {}".format(th))


plt.legend()
plt.tight_layout()
plt.show()
# %%
