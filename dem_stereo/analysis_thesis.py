# %%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
csv_path = "result_experiment/experiment_006.csv"
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
    plt.ylim(
        50,
        100,
    )
    plt.xlim(1, 8)
    plt.title("EVENT_TH: {}".format(th))
    plt.xlabel("Input Time Step", fontsize=18)
    plt.ylabel("Accuracy [%]", fontsize=18)
    plt.tick_params(labelsize=15)
    plt.legend()
    # plt.rcParams["font.size"] = 180


plt.tight_layout()
plt.savefig("result_two.pdf")
plt.savefig("result_two.png")
plt.show()
# %%
csv_path = "result_experiment/experiment_006.csv"
data = pd.read_csv(csv_path)


# accuracy = []
event_th = [0.15]
time_change_lst = [False, True]
# time_change_lst = [False]
plt.figure(figsize=(5, 5))
for i, acc_time in enumerate(data["ACCUMULATE_EVENT_MILITIME"].unique()):
    print(acc_time)
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
                    & (data["TIME_CHANGE"] == False)
                    & (data["ACCUMULATE_EVENT_MILITIME"] == acc_time),
                    "Precision",
                ].values[0]
            )
            recall.append(
                data.loc[
                    (data["EVENT_TH"] == th)
                    & (data["FINISH_STEP"] == fin_step)
                    & (data["TIME_CHANGE"] == False)
                    & (data["ACCUMULATE_EVENT_MILITIME"] == acc_time),
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
                    & (data["TIME_CHANGE"] == True)
                    & (data["ACCUMULATE_EVENT_MILITIME"] == acc_time),
                    "Precision",
                ].values[0]
            )
            recall_time.append(
                data.loc[
                    (data["EVENT_TH"] == th)
                    & (data["FINISH_STEP"] == fin_step)
                    & (data["TIME_CHANGE"] == True)
                    & (data["ACCUMULATE_EVENT_MILITIME"] == acc_time),
                    "Recall",
                ].values[0]
            )
        # plt_width = len(data["EVENT_TH"].unique())
        plt_width = len(event_th)
        plt_width = len(data["ACCUMULATE_EVENT_MILITIME"].unique())
        x_lst = data["FINISH_STEP"].unique()
        x_lst_change = np.array(data["FINISH_STEP"].unique())
        x_lst_change = x_lst_change[:-1]
        print(recall_time)
        plt.subplot(1, plt_width, i + 1)
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
        plt.ylim(
            50,
            100,
        )
        plt.xlim(1, 8)
        # plt.title("EVENT_TH: {}".format(th))
        # plt.title("acc_time: {}".format(acc_time))
        plt.title(f"two cameras({acc_time}ms))")
        plt.xlabel("Input Time Step", fontsize=18)
        plt.ylabel("Accuracy [%]", fontsize=18)
        plt.tick_params(labelsize=15)
        plt.legend()
        # plt.rcParams["font.size"] = 180

plt.tight_layout()
plt.savefig("result_two.pdf")
plt.savefig("result_two.png")
plt.savefig("result_two.svg")
plt.show()

# %%
