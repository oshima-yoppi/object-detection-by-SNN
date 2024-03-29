import os
from .const import *


def v2e_cmd(save_path, video_path, raw_event_dir, skip_video_output=True):
    # file_name = os.path.join(raw_event_path, f"{number_}.h5")
    # video_file_name = os.path.join(video_path, f"{number_}.avi")
    cmd = f"python ../v2e-master/v2e.py -i {video_path} -o {raw_event_dir} --overwrite --output_file_path {save_path}  --disable --dvs346 --neg_thres {EVENT_TH} --pos_thres {EVENT_TH} --dvs_aedat2 None --dvs_text None --no_preview --dvs_h5 True"
    if skip_video_output:
        cmd += " --skip_video_output"

    os.system(cmd)
