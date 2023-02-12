import os
from .const import * 
def v2e_cmd(number):


    number_ = str(number).zfill(5)
    file_name = os.path.join(RAW_EVENT_PATH, f'{number_}.h5')
    cmd = f"python ../v2e-master/v2e.py -i blender/video/{number_}.avi -o {RAW_EVENT_PATH} --overwrite --output_file_path {file_name}  --disable --dvs346 --neg_thres {EVENT_TH} --pos_thres {EVENT_TH} --dvs_aedat2 None --dvs_text None --skip_video_output --no_preview --dvs_h5 True"


    os.system(cmd)