import os 
import itertools

soft_reset_lst = [True]
parm_learn_lst = [True]
finish_step_lst = [4, 8]
accumulate_event_militime_lst = [10, 20 , 40, 60, 80, 100]
combinations = list(itertools.product(soft_reset_lst, parm_learn_lst, finish_step_lst, accumulate_event_militime_lst))

for soft_reset, parm_learn, finish_step, accumulate_event_militime in combinations:
    cmd = "python experiment.py"
    if soft_reset:
        cmd += " --soft_reset"
    if parm_learn:
        cmd += " --parm_learn"
    cmd += f" --FINISH_STEP {finish_step}"
    cmd += f" --ACCUMULATE_EVENT_MILITIME {accumulate_event_militime}"
    os.system(cmd)
