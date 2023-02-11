import os
from .const import * 
def v2e_cmd(number):
    output_dir = RAW_EVENT_PATH
    output_dir_th = os.path.join(output_dir, EVENT_TH)


    number_ = str(number).zfill(5)
    cmd = f"python ../v2e-master/v2e.py -i blender/video/{number_}.avi -o {output_dir_th} --overwrite --output_file_path data/{number_}.h5  --disable --dvs346 --neg_thres 0.1 --pos_thres 0.1 --dvs_aedat2 None --dvs_text None --skip_video_output --no_preview --dvs_h5 True"


    os.system(cmd)