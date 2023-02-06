import os
cmd = "python ../v2e-master/v2e.py --output_in_place ../v2e-master/outputs/exp --disable --dvs346 --neg_thres 0.1 --pos_thres 0.1 --dvs_exposure duration 0.05"
cmd = "python ../v2e-master/v2e.py -o data/dvs --unique_output_folder  --disable --dvs346 --neg_thres 0.1 --pos_thres 0.1 --dvs_exposure duration 0.05"
cmd = 
os.system(cmd)