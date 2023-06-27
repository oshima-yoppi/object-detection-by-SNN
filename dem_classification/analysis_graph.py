#%%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
#%%
csv_path = "result_experiment\experiment_008.csv"
data = pd.read_csv(csv_path)

finish_step = 4
event_counts = [True, False]
# event_counts = [False]
linestyles = ['solid', 'dashed']

for (event_count, linestyle) in zip(event_counts, linestyles):
    x = data.loc[(data["FINISH_STEP"] == finish_step ) & (data["EVENT_COUNT"] == event_count), "ACCUMULATE_EVENT_MILITIME"]
    y_precision = data.loc[(data["FINISH_STEP"] == finish_step ) & (data["EVENT_COUNT"] == event_count), "Precision"]
    y_recall = data.loc[(data["FINISH_STEP"] == finish_step ) & (data["EVENT_COUNT"] == event_count), "Recall"]
    y_iou = data.loc[(data["FINISH_STEP"] == finish_step ) & (data["EVENT_COUNT"] == event_count), "IoU"]
    # y_accuracy = data.loc[(data["FINISH_STEP"] == finish_step ) & (data["EVENT_COUNT"] == event_count), "Accuracy"]


    plt.plot(x, y_precision, label=f"Precision ({event_count})", linestyle=linestyle, marker="x", color="red")
    plt.plot(x, y_recall, label=f"Recall ({event_count})", linestyle=linestyle, marker="x", color="blue")
    plt.plot(x, y_iou, label=f"IoU ({event_count})", linestyle=linestyle, marker="x", color="orange")
    # plt.plot(x, y_accuracy, label=f"Accuracy ({event_count})", linestyle=linestyle, marker="x", color="green")

plt.legend()
plt.ylim(0, 100)
plt.xlabel(f"Accumulate event time [ms] (per step)")
plt.title(f"Finish step: {finish_step}")
plt.show()


#%%
# plt.plot(x, y_precision, label="Precision", marker="x")
# plt.plot(x, y_recall, label="Recall", marker="x")
# plt.plot(x, y_accuracy, label="Accuracy" ,  marker="x")

# %%
