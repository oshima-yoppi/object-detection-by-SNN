import os
from .const import *


def v2e_cmd(number, event_th, raw_event_path=RAW_EVENT_PATH, video_path=VIDEO_PATH):

    number_ = str(number).zfill(5)
    file_name = os.path.join(raw_event_path, f"{number_}.h5")
    video_file_name = os.path.join(video_path, f"{number_}.avi")
    cmd = f"python ../v2e-master/v2e.py -i {video_file_name} -o {raw_event_path} --overwrite --output_file_path {file_name}  --disable --dvs346 --neg_thres {event_th} --pos_thres {event_th} --dvs_aedat2 None --dvs_text None --skip_video_output --no_preview --dvs_h5 True"

    os.system(cmd)
