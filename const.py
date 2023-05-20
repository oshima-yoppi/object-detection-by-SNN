import json

with open('const.json') as file:
    constants = json.load(file)

constants["MY_CONSTANT"] = 24


print(constants["MY_CONSTANT"])  # 24