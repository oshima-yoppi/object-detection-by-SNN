# %%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
csv_path = "result_experiment/experiment_030.csv"
data = pd.read_csv(csv_path)

# plt.figure()
repeat_input_lst = data.loc[:, "REPEAT_INPUT"].unique()
finish_step_lst = data.loc[:, "FINISH_STEP"].unique()
leaky_lst = data.loc[:, "BETA"].unique()
repeat_input_lst.sort()
leaky_lst.sort()
finish_step_lst.sort()

for repeat_input in repeat_input_lst:
    plt.figure()
    for j, time_step in enumerate(finish_step_lst):
        precision = []
        recall = []
        iou = []
        f_measure = []
        for leaky in leaky_lst:
            precision.append(
                data.loc[
                    (data["FINISH_STEP"] == time_step)
                    & (data["BETA"] == leaky)
                    & (data["REPEAT_INPUT"] == repeat_input),
                    "Precision",
                ].values[0]
            )
            recall.append(
                data.loc[
                    (data["FINISH_STEP"] == time_step)
                    & (data["BETA"] == leaky)
                    & (data["REPEAT_INPUT"] == repeat_input),
                    "Recall",
                ].values[0]
            )
            iou.append(
                data.loc[
                    (data["FINISH_STEP"] == time_step)
                    & (data["BETA"] == leaky)
                    & (data["REPEAT_INPUT"] == repeat_input),
                    "IoU",
                ].values[0]
            )
            f_measure.append(
                data.loc[
                    (data["FINISH_STEP"] == time_step)
                    & (data["BETA"] == leaky)
                    & (data["REPEAT_INPUT"] == repeat_input),
                    "F-Measure",
                ].values[0]
            )
        plt.subplot(1, len(finish_step_lst), j + 1)
        plt.plot(leaky_lst, precision, label="Precision", marker="x")
        plt.plot(leaky_lst, recall, label="Recall", marker="x")
        plt.plot(leaky_lst, iou, label="IoU", marker="x")
        plt.plot(leaky_lst, f_measure, label="F-Measure", marker="x")
        plt.ylim(70, 100)
        plt.legend()
        plt.xlabel("Leaky")
        plt.ylabel("Accuracy[%]]")
        plt.title(f"when timestep is {time_step} and repeat input is {repeat_input}")

plt.show()
# for step in finish_step_lst:
#     precision.append(
#         data.loc[
#             (data["FINISH_STEP"] == step) & (data["BETA"] == leaky), "Precision"
#         ].values[0]
#     )
#     recall.append(
#         data.loc[
#             (data["FINISH_STEP"] == step) & (data["BETA"] == leaky), "Recall"
#         ].values[0]
#     )
#     iou.append(
#         data.loc[(data["FINISH_STEP"] == step) & (data["BETA"] == leaky), "IoU"].values[
#             0
#         ]
#     )
#     f_measure.append(
#         data.loc[
#             (data["FINISH_STEP"] == step) & (data["BETA"] == leaky), "F-Measure"
#         ].values[0]
#     )

plt.plot(finish_step_lst, precision, label="Precision", marker="x")
plt.plot(finish_step_lst, recall, label="Recall", marker="x")
plt.plot(finish_step_lst, iou, label="IoU", marker="x")
plt.plot(finish_step_lst, f_measure, label="F-Measure", marker="x")
plt.legend()
plt.xlabel("Finish Step")
plt.ylabel("Score")
plt.title("Score vs Finish Step")
plt.show()
