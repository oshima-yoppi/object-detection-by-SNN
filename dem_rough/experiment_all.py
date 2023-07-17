import os
import itertools


csv_dir = "result_experiment"
prefix = "experiment_"

file_count = len(os.listdir(csv_dir))
file_count += 1

csv_name = prefix + str(file_count).zfill(3) + ".csv"
csv_path = os.path.join(csv_dir, csv_name)

soft_reset_lst = [True]
parm_learn_lst = [False]
# finish_step_lst = [2, 5]
finish_step_lst = [2, 5, 8]
# accumulate_event_militime_lst = [10, 40, 80]
accumulate_event_militime_lst = [80]
# event_count = [False, True]
event_count_lst = [False]
evnet_th_lst = [0.05, 0.1, 0.15]
time_change_lst = [False, True]
# accumulate_event_militime_lst = [40, 60, 80]
combinations = list(
    itertools.product(soft_reset_lst, parm_learn_lst, finish_step_lst, accumulate_event_militime_lst, event_count_lst, evnet_th_lst, time_change_lst)
)

for (
    soft_reset,
    parm_learn,
    finish_step,
    accumulate_event_militime,
    event_count,
    evnet_th,
    time_change,
) in combinations:
    cmd = "python experiment.py"
    if soft_reset:
        cmd += " --soft_reset"
    if parm_learn:
        cmd += " --PARM_LEARN"
    cmd += f" --FINISH_STEP {finish_step}"
    cmd += f" --ACCUMULATE_EVENT_MILITIME {accumulate_event_militime}"
    cmd += f" --CSV_PATH {csv_path}"
    cmd += f" --EVENT_TH {evnet_th}"
    if event_count:
        cmd += " --EVENT_COUNT"
    if time_change:
        cmd += " --TIME_CHANGE"
    os.system(cmd)
