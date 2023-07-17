import argparse
import json
import csv
import os


# const.jsonの中身が変更される
def update_constant(args):
    # 変更したデータを保存
    constants = {
        "soft_reset": args.soft_reset,
        "PARM_LEARN": args.PARM_LEARN,
        "FINISH_STEP": args.FINISH_STEP,
        "ACCUMULATE_EVENT_MILITIME": args.ACCUMULATE_EVENT_MILITIME,
        "EVENT_COUNT": args.EVENT_COUNT,
        "EVENT_TH": args.EVENT_TH,
        "TIME_CHANGE": args.TIME_CHANGE,
    }
    with open("module/const_base.json", "w") as file:
        json.dump(constants, file)


def log_experiment(constants, results):
    with open("experiment_log.txt", "a") as file:
        file.write("-------------\n")
        file.write("conditions\n")
        for key, value in constants.items():
            file.write(f"\t{key}: {value}\n")

        file.write("results\n")
        for key, value in results.items():
            file.write(f"\t{key}: {value}\n")
        file.write("-------------\n")


def check_csv_file(filne_name):
    with open(filne_name, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        if header is None:
            return False
        return True


def write_csv(constants, results, csv_file):
    # カラム名と値のリストを作成
    columns = list(constants.keys()) + list(results.keys())
    values = []
    for key in columns:
        if key in constants:
            values.append(constants[key])
        else:
            values.append(results[key])
    # values = list(constants.values()) + list(results.values())

    # # CSVファイルにデータを保存
    # with open(csv_file, 'a', newline='') as file:
    #     writer = csv.writer(file)
    #     # if check_csv_file(csv_file) == False:
    #     writer.writerow(columns)
    #     writer.writerow(values)
    # CSVファイルが存在しない場合のみカラム名を書き込む
    if not os.path.isfile(csv_file):
        with open(csv_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)

    # データを書き込む
    with open(csv_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(values)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--soft_reset", action="store_true"
)  # store_true: 引数があればTrue, なければFalse.これしないと正しく出ない。https://qiita.com/hirorin/items/fbcf76c1119da24e2eeb
parser.add_argument("--PARM_LEARN", action="store_true")
parser.add_argument("--FINISH_STEP", type=int)
parser.add_argument("--ACCUMULATE_EVENT_MILITIME", type=int)
parser.add_argument("--CSV_PATH", type=str)
parser.add_argument("--EVENT_COUNT", action="store_true")
parser.add_argument("--EVENT_TH", type=float)
parser.add_argument("--TIME_CHANGE", type=float)
args = parser.parse_args()
print(args.soft_reset)
CSV_PATH = args.CSV_PATH
delattr(args, "CSV_PATH")
update_constant(args)

import train

hist = train.main()

import analysis

results = analysis.main(hist=hist)


log_experiment(vars(args), results)
write_csv(vars(args), results, CSV_PATH)
