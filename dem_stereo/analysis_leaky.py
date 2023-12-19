# %%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
csv_path = "result_experiment/experiment_027.csv"
data = pd.read_csv(csv_path)


# beta_lst = [0.2, 0.4, 0.6, 0.8, 1.0]
beta_lst = data["BETA"].unique()
beta_lst.sort()
# finish_step_lst = [1, 2, 4, 5, 6, 8]

# finish_step_lst = [1, 2, 4, 5, 6, 8]
finish_step_lst = [8]
thres_learn_lst = [False, True]

for thres_learn in thres_learn_lst:
    plt.figure(figsize=(5, 5))
    for i, fin_step in enumerate(finish_step_lst):
        precision = []
        recall = []
        iou = []
        f_measure = []
        for beta in beta_lst:
            precision.append(
                data.loc[
                    (data["BETA"] == beta)
                    & (data["FINISH_STEP"] == fin_step)
                    & (data["THRESHOLD_LEARN"] == thres_learn),
                    "Precision",
                ].values[0]
            )
            recall.append(
                data.loc[
                    (data["BETA"] == beta)
                    & (data["FINISH_STEP"] == fin_step)
                    & (data["THRESHOLD_LEARN"] == thres_learn),
                    "Recall",
                ].values[0]
            )
            iou.append(
                data.loc[
                    (data["BETA"] == beta)
                    & (data["FINISH_STEP"] == fin_step)
                    & (data["THRESHOLD_LEARN"] == thres_learn),
                    "IoU",
                ].values[0]
            )
            f_measure.append(
                data.loc[
                    (data["BETA"] == beta)
                    & (data["FINISH_STEP"] == fin_step)
                    & (data["THRESHOLD_LEARN"] == thres_learn),
                    "F-Measure",
                ].values[0]
            )
        plt_width = len(finish_step_lst)
        plt.subplot(1, plt_width, i + 1)
        plt.plot(
            beta_lst,
            precision,
            label="Precision",
            linestyle="dashed",
            marker="x",
            color="red",
        )
        plt.plot(
            beta_lst,
            recall,
            label="Recall",
            linestyle="dashed",
            marker="x",
            color="blue",
        )
        plt.plot(
            beta_lst,
            iou,
            label="IOU",
            linestyle="dashed",
            marker="x",
            color="green",
        )
        plt.plot(
            beta_lst,
            f_measure,
            label="F-measure",
            linestyle="dashed",
            marker="x",
            color="black",
        )
        plt.ylim(
            50,
            100,
        )
        plt.xlim(0.2, 1.0)
        plt.title("FINISH_STEP: {}".format(fin_step))
        plt.xlabel("Leaky β", fontsize=18)
        plt.ylabel("Accuracy [%]", fontsize=18)
        plt.tick_params(labelsize=15)
        plt.legend()
    plt.tight_layout()
    plt.show()
