import os 
import itertools

soft_reset_lst = [True, False]
parm_learn_lst = [True, False]
finish_step_lst = [1 , 5, 9]
combinations = list(itertools.product(soft_reset_lst, parm_learn_lst, finish_step_lst))

for soft_reset, parm_learn, finish_step in combinations:
    cmd = "python experiment.py"
    if soft_reset:
        cmd += " --soft_reset"
    if parm_learn:
        cmd += " --parm_learn"
    cmd += f" --FINISH_STEP {finish_step}"
    os.system(cmd)
