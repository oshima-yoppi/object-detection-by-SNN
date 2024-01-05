from module.cmd import v2e_cmd
from module.const import *
import os

if __name__ == "__main__":
    while 1:
        number = int(input("number: "))
        left_video_path = os.path.join(VIDEO_LEFT_PATH, f"{str(number).zfill(5)}.avi")
        right_video_path = os.path.join(VIDEO_RIGHT_PATH, f"{str(number).zfill(5)}.avi")
        save_dir = "check_event_video"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_left_path = os.path.join(save_dir, f"{str(number).zfill(5)}_left.avi")
        save_right_path = os.path.join(save_dir, f"{str(number).zfill(5)}_right.avi")
        v2e_cmd(
            save_path=save_left_path,
            video_path=left_video_path,
            raw_event_dir=save_dir,
            skip_video_output=False,
        )
        v2e_cmd(
            save_path=save_right_path,
            video_path=right_video_path,
            raw_event_dir=save_dir,
            skip_video_output=False,
        )
