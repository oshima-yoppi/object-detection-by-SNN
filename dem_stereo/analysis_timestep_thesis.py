# %%import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# %%
csv_path = "result_experiment/experiment_029.csv"
data = pd.read_csv(csv_path)

# finish_step_lst = [2, 4, 6, 8]
finish_step_lst = data.loc[:, "FINISH_STEP"].unique()
# leaky_lst = data.loc[:, "BETA"].unique()
repeat_input_lst = data.loc[:, "REPEAT_INPUT"].unique()
leaky = 1
# save path
result_save_dir = "thesis_result"
os.makedirs(result_save_dir, exist_ok=True)


# リピートインプットありの場合
plt.figure()
precision = []
recall = []
iou = []
f_measure = []
bool_repeat_input = True
for timestep in finish_step_lst:
    precision.append(
        data.loc[
            (data["FINISH_STEP"] == timestep)
            & (data["BETA"] == leaky)
            & (data["REPEAT_INPUT"] == bool_repeat_input),
            "Precision",
        ].values[0]
    )
    recall.append(
        data.loc[
            (data["FINISH_STEP"] == timestep)
            & (data["BETA"] == leaky)
            & (data["REPEAT_INPUT"] == bool_repeat_input),
            "Recall",
        ].values[0]
    )
    iou.append(
        data.loc[
            (data["FINISH_STEP"] == timestep)
            & (data["BETA"] == leaky)
            & (data["REPEAT_INPUT"] == bool_repeat_input),
            "IoU",
        ].values[0]
    )
    f_measure.append(
        data.loc[
            (data["FINISH_STEP"] == timestep)
            & (data["BETA"] == leaky)
            & (data["REPEAT_INPUT"] == bool_repeat_input),
            "F-Measure",
        ].values[0]
    )
plt.plot(finish_step_lst, precision, label="Precision", marker="x")
plt.plot(finish_step_lst, recall, label="Recall", marker="x")
plt.plot(finish_step_lst, iou, label="IoU", marker="x")
plt.plot(finish_step_lst, f_measure, label="F-Measure", marker="x")
plt.ylim(75, 100)
plt.xlabel("time step")
plt.ylabel("accuracy [%]")
plt.title(f"Repeat input: {bool_repeat_input}")
plt.legend()
plt.savefig(os.path.join(result_save_dir, "repeat_input_true.png"))
plt.savefig(os.path.join(result_save_dir, "repeat_input_true.pdf"))
plt.show()


# リピートインプットなしの場合
plt.figure()
precision = []
recall = []
iou = []
f_measure = []
bool_repeat_input = False
for timestep in finish_step_lst:
    precision.append(
        data.loc[
            (data["FINISH_STEP"] == timestep)
            & (data["BETA"] == leaky)
            & (data["REPEAT_INPUT"] == bool_repeat_input),
            "Precision",
        ].values[0]
    )
    recall.append(
        data.loc[
            (data["FINISH_STEP"] == timestep)
            & (data["BETA"] == leaky)
            & (data["REPEAT_INPUT"] == bool_repeat_input),
            "Recall",
        ].values[0]
    )
    iou.append(
        data.loc[
            (data["FINISH_STEP"] == timestep)
            & (data["BETA"] == leaky)
            & (data["REPEAT_INPUT"] == bool_repeat_input),
            "IoU",
        ].values[0]
    )
    f_measure.append(
        data.loc[
            (data["FINISH_STEP"] == timestep)
            & (data["BETA"] == leaky)
            & (data["REPEAT_INPUT"] == bool_repeat_input),
            "F-Measure",
        ].values[0]
    )
plt.plot(finish_step_lst, precision, label="Precision", marker="x")
plt.plot(finish_step_lst, recall, label="Recall", marker="x")
plt.plot(finish_step_lst, iou, label="IoU", marker="x")
plt.plot(finish_step_lst, f_measure, label="F-Measure", marker="x")
plt.ylim(75, 100)
plt.xlabel("time step")
plt.ylabel("accuracy [%]")
plt.title(f"Repeat input: {bool_repeat_input}")
plt.legend()
plt.savefig(os.path.join(result_save_dir, "repeat_input_false.png"))
plt.savefig(os.path.join(result_save_dir, "repeat_input_false.pdf"))
plt.show()
