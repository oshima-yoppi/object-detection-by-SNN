import os 
import itertools


csv_dir = "result_experiment"
prefix = "experiment_"

file_count = len(os.listdir(csv_dir))
file_count += 1
csv_name = prefix + str(file_count).zfill(3) + ".csv"
csv_path = os.path.join(csv_dir, csv_name)

soft_reset_lst = [True, False]
parm_learn_lst = [True, False]
finish_step_lst = [4, 8]
accumulate_event_militime_lst = [10, 20 , 40, 80, 100]
# accumulate_event_militime_lst = [40, 60, 80]
combinations = list(itertools.product(soft_reset_lst, parm_learn_lst, finish_step_lst, accumulate_event_militime_lst))

for soft_reset, parm_learn, finish_step, accumulate_event_militime in combinations:
    cmd = "python experiment.py"
    if soft_reset:
        cmd += " --soft_reset"
    if parm_learn:
        cmd += " --PARM_LEARN"
    cmd += f" --FINISH_STEP {finish_step}"
    cmd += f" --ACCUMULATE_EVENT_MILITIME {accumulate_event_militime}"
    cmd += f" --CSV_PATH {csv_path}"
    os.system(cmd)
