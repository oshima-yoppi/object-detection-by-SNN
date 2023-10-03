# %%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
csv_path = "result_experiment/experiment_017.csv"
data = pd.read_csv(csv_path)


# accuracy = []
event_th = [0.15]
time_change_lst = [False, True]
# time_change_lst = [False]
plt.figure(figsize=(5, 5))
for j, th in enumerate(event_th):
    precision = []
    recall = []
    precision_time = []
    recall_time = []
    for fin_step in data["FINISH_STEP"].unique():
        precision.append(
            data.loc[
                (data["EVENT_TH"] == th)
                & (data["FINISH_STEP"] == fin_step)
                & (data["TIME_CHANGE"] == False),
                "Precision",
            ].values[0]
        )
        recall.append(
            data.loc[
                (data["EVENT_TH"] == th)
                & (data["FINISH_STEP"] == fin_step)
                & (data["TIME_CHANGE"] == False),
                "Recall",
            ].values[0]
        )
        # print(type(fin_step))
        if fin_step == 8:
            continue
        precision_time.append(
            data.loc[
                (data["EVENT_TH"] == th)
                & (data["FINISH_STEP"] == fin_step)
                & (data["TIME_CHANGE"] == True),
                "Precision",
            ].values[0]
        )
        recall_time.append(
            data.loc[
                (data["EVENT_TH"] == th)
                & (data["FINISH_STEP"] == fin_step)
                & (data["TIME_CHANGE"] == True),
                "Recall",
            ].values[0]
        )
    plt_width = len(data["EVENT_TH"].unique())
    plt_width = len(event_th)
    x_lst = data["FINISH_STEP"].unique()
    x_lst_change = np.array(data["FINISH_STEP"].unique())
    x_lst_change = x_lst_change[:-1]
    print(recall_time)
    plt.subplot(1, plt_width, j + 1)
    plt.plot(
        data["FINISH_STEP"].unique(),
        precision,
        label="Precision",
        linestyle="dashed",
        marker="x",
        color="red",
    )
    plt.plot(
        data["FINISH_STEP"].unique(),
        recall,
        label="Recall",
        linestyle="dashed",
        marker="x",
        color="blue",
    )
    plt.plot(
        x_lst_change,
        precision_time,
        label="Precision (Proposed)",
        linestyle="solid",
        marker="x",
        color="red",
    )
    plt.plot(
        x_lst_change,
        recall_time,
        label="Recall (Proposed)",
        linestyle="solid",
        marker="x",
        color="blue",
    )
    plt.ylim(50, 100)
    plt.xlim(1, 8)
    plt.title("one camera")
    plt.xlabel("Input Time Step", fontsize=18)
    plt.ylabel("Accuracy [%]", fontsize=18)
    plt.tick_params(labelsize=15)
    plt.legend()
    # plt.rcParams["font.size"] = 180


plt.tight_layout()
plt.savefig("result_one.pdf")
plt.savefig("result_one.png")
plt.savefig("result_one.svg")
plt.show()
# %%
