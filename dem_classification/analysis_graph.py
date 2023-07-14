#%%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

#%%
csv_path = "result_experiment\experiment_008.csv"
data = pd.read_csv(csv_path)

finish_step = 4
event_counts = [True, False]
# event_counts = [False]
linestyles = ["solid", "dashed"]

for (event_count, linestyle) in zip(event_counts, linestyles):
    x = data.loc[
        (data["FINISH_STEP"] == finish_step) & (data["EVENT_COUNT"] == event_count),
        "ACCUMULATE_EVENT_MILITIME",
    ]
    y_precision = data.loc[
        (data["FINISH_STEP"] == finish_step) & (data["EVENT_COUNT"] == event_count),
        "Precision",
    ]
    y_recall = data.loc[
        (data["FINISH_STEP"] == finish_step) & (data["EVENT_COUNT"] == event_count),
        "Recall",
    ]
    y_iou = data.loc[
        (data["FINISH_STEP"] == finish_step) & (data["EVENT_COUNT"] == event_count),
        "IoU",
    ]
    # y_accuracy = data.loc[(data["FINISH_STEP"] == finish_step ) & (data["EVENT_COUNT"] == event_count), "Accuracy"]

    plt.plot(
        x,
        y_precision,
        label=f"Precision ({event_count})",
        linestyle=linestyle,
        marker="x",
        color="red",
    )
    plt.plot(
        x,
        y_recall,
        label=f"Recall ({event_count})",
        linestyle=linestyle,
        marker="x",
        color="blue",
    )
    plt.plot(
        x,
        y_iou,
        label=f"IoU ({event_count})",
        linestyle=linestyle,
        marker="x",
        color="orange",
    )
    # plt.plot(x, y_accuracy, label=f"Accuracy ({event_count})", linestyle=linestyle, marker="x", color="green")

plt.legend()
plt.ylim(0, 100)
plt.xlabel(f"Accumulate event time [ms] (per step)")
plt.title(f"Finish step: {finish_step}")
plt.show()


#%%
csv_path = "result_experiment\experiment_011.csv"
data = pd.read_csv(csv_path)

precision = []
recall = []
accuracy = []
event_th = [0.05, 0.1, 0.15]
for th in event_th:
    precision.append(data.loc[(data["EVENT_TH"] == th), "Precision"].values[0])
    recall.append(data.loc[(data["EVENT_TH"] == th), "Recall"].values[0])
    accuracy.append(data.loc[(data["EVENT_TH"] == th), "Accuracy"].values[0])

plt.plot(
    event_th, precision, label="Precision", linestyle="solid", marker="x", color="red"
)
plt.plot(event_th, recall, label="Recall", linestyle="solid", marker="x", color="blue")
plt.plot(
    event_th, accuracy, label="Accuracy", linestyle="solid", marker="x", color="green"
)
plt.legend()
plt.show()
# %%
