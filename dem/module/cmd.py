import os

def v2e_cmd(number):
    cmd = f"python ../v2e-master/v2e.py -i blender/video/{number}.avi -o data --overwrite --output_file_path data/{number}.h5  --disable --dvs346 --neg_thres 0.1 --pos_thres 0.1 --dvs_exposure duration 0.1 --dvs_aedat2 None --dvs_text None --skip_video_output --no_preview --dvs_h5 True"


    cmd = f"python ../v2e-master/v2e.py -i blender/video/{number}.avi -o data --overwrite --output_file_path data/{number}.h5  --disable --dvs346 --neg_thres 0.1 --pos_thres 0.1 --dvs_aedat2 None --dvs_text None --skip_video_output --no_preview --dvs_h5 True"


    os.system(cmd)