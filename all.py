import json
import importlib
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--time', type=int, default=100)

args = parser.parse_args()
def vary(time):
    # 変更したデータを保存
    constants = {
        "time": time
    }
    with open('const.json', 'w') as file:
        json.dump(constants, file)


# if __name__ == "__main__":
vary(args.time)
import a