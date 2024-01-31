import os
import itertools
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--leaky",
#     "-l",
#     required=True,
#     nargs="*",
#     type=float,
# )
# args = parser.parse_args()
# L = args.leaky


csv_dir = "result_experiment"
prefix = "experiment_"

file_count = len(os.listdir(csv_dir))
file_count += 1
if os.path.exists(csv_dir) is False:
    os.mkdir(csv_dir)
csv_name = prefix + str(file_count).zfill(3) + ".csv"
csv_path = os.path.join(csv_dir, csv_name)
with open(csv_path, "w") as f:
    f.write("")

soft_reset_lst = [True]
finish_step_lst = [2, 3, 4, 6]
# accumulate_event_militime_lst = [10, 40, 80]
# accumulate_event_militime_lst = [50, 100]
accumulate_event_militime_lst = [100]
# event_count_lst = [False, True]
event_count_lst = [False]
evnet_th_lst = [0.15]
# time_change_lst = [False, True]
time_change_lst = [False]  ##############################
beta_learn_lst = [False]
threshold_learn_lst = [False]
beta_lst = [0.25, 0.5, 0.75, 1.0]
repeat_input_lst = [False]
threshold_lst = [1]
# time_aware_loss_lst = [False, True]
time_aware_loss_lst = [True]
combinations = list(
    itertools.product(
        soft_reset_lst,
        finish_step_lst,
        accumulate_event_militime_lst,
        event_count_lst,
        evnet_th_lst,
        time_change_lst,
        beta_learn_lst,
        threshold_learn_lst,
        beta_lst,
        repeat_input_lst,
        time_aware_loss_lst,
        threshold_lst,
    )
)
cc = 0
for (
    soft_reset,
    finish_step,
    accumulate_event_militime,
    event_count,
    evnet_th,
    time_change,
    beta_learn,
    threshold_learn,
    beta,
    repeat_input,
    time_aware_loss,
    threshold,
) in combinations:
    # if cc <=14:
    #     cc += 1
    #     continue
    cmd = "python experiment.py"
    if soft_reset:
        cmd += " --soft_reset"
    cmd += f" --FINISH_STEP {finish_step}"
    cmd += f" --ACCUMULATE_EVENT_MILITIME {accumulate_event_militime}"
    cmd += f" --CSV_PATH {csv_path}"
    cmd += f" --EVENT_TH {evnet_th}"
    if event_count:
        cmd += " --EVENT_COUNT"
    if time_change:
        cmd += " --TIME_CHANGE"
    if beta_learn:
        cmd += " --BETA_LEARN"
    if threshold_learn:
        cmd += " --THRESHOLD_LEARN"
    cmd += f" --BETA {beta}"
    if repeat_input:
        cmd += " --REPEAT_INPUT"
    if time_aware_loss:
        cmd += " --TIME_AWARE_LOSS"
    cmd += f" --THRESHOLD {threshold}"

    os.system(cmd)
