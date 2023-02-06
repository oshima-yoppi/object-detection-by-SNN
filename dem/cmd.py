import os
cmd = "python v2e.py --output_in_place ../v2e-master/outputs/exp --disable --dvs346 --neg_thres 0.1 --pos_thres 0.1 --dvs_exposure duration 0.05"
cmd = "python ../v2e-master/v2e.py --output_file_path ../dem/data/a.h5  --disable --dvs346 --neg_thres 0.1 --pos_thres 0.1 --dvs_exposure duration 0.05 --dvs_aedat2 None --dvs_text None"
cmd = "python ../v2e-master/v2e.py -o data --overwrite --output_file_path data/a.h5  --disable --dvs346 --neg_thres 0.1 --pos_thres 0.1 --dvs_exposure duration 0.05 --dvs_aedat2 None --dvs_text None --skip_video_output --no_preview --dvs_h5 True"

os.system(cmd)