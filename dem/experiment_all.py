import os 

cmd = "python experiment.py --soft_reset True --parm_learn True --FINISH_STEP 8"
os.system(cmd)


cmd = "python experiment.py --soft_reset Falase --parm_learn True --FINISH_STEP 8"
os.system(cmd)
