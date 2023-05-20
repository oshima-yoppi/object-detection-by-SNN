import json
print('import a')
with open('const.json') as file:
    constants = json.load(file)

TIME = constants['time']
print(TIME)


