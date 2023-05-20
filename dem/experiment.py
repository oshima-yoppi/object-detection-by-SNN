import argparse
import json
import train
# const.jsonの中身が変更される
def update_constant(args):
    # 変更したデータを保存
    constants = {
        "soft_reset": args.soft_reset,
        "parm_learn": args.parm_learn,
        "FINISH_STEP": args.finish_step
    }
    with open('const.json', 'w') as file:
        json.dump(constants, file)

parser = argparse.ArgumentParser()
parser.add_argument('--soft_reset', type=bool)
parser.add_argument('--parm_learn', type=bool)
parser.add_argument('--FINISH_STEP', type=int)

args = parser.parse_args()
update_constant(args)
train.main()



