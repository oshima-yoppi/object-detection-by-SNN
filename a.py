import json

with open('const.json') as file:
    constants = json.load(file)

constants["MY_CONSTANT"] = 24

# 変更したデータを保存
with open('const.json', 'w') as file:
    json.dump(constants, file)