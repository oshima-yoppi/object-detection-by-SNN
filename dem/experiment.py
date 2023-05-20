import argparse
import json
# const.jsonの中身が変更される
def update_constant(args):
    # 変更したデータを保存
    constants = {
        "soft_reset": args.soft_reset,
        "parm_learn": args.parm_learn,
        "FINISH_STEP": args.FINISH_STEP
    }
    with open('module/const_base.json', 'w') as file:
        json.dump(constants, file)
def log_experiment(constants, results):
    with open('experiment_log.txt', 'a') as file:
        file.write("-------------\n")
        file.write("conditions\n")
        for key, value in constants.items():
            file.write(f"{key}: {value}\n")

        file.write("results\n")
        for key, value in results.items():
            file.write(f"{key}: {value}\n")
        file.write("-------------\n")


parser = argparse.ArgumentParser()
parser.add_argument('--soft_reset', action='store_true') # store_true: 引数があればTrue, なければFalse.これしないと正しく出ない。https://qiita.com/hirorin/items/fbcf76c1119da24e2eeb
parser.add_argument('--parm_learn', action='store_true')
parser.add_argument('--FINISH_STEP', type=int)

args = parser.parse_args()
print(args.soft_reset)
update_constant(args)

import train
train.main()

import analysis
results = analysis.main()
log_experiment(vars(args), results)



