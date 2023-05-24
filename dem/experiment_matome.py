#%%
import re
import pandas as pd

data = '''
-------------
conditions
	soft_reset: True
	PARM_LEARN: True
	FINISH_STEP: 4
	ACCUMULATE_EVENT_MILITIME: 10
results
	IoU: 0.3000732604778993
	Energy per inference: 4.979087862011511e-07
	Spike Rate: 0.1255052089691162
-------------
-------------
conditions
	soft_reset: True
	PARM_LEARN: True
	FINISH_STEP: 4
	ACCUMULATE_EVENT_MILITIME: 20
results
	IoU: 0.45900729780395827
	Energy per inference: 6.193101853568805e-07
	Spike Rate: 0.15610621869564056
-------------
-------------
conditions
	soft_reset: True
	PARM_LEARN: True
	FINISH_STEP: 4
	ACCUMULATE_EVENT_MILITIME: 40
results
	IoU: 0.6486806876957416
	Energy per inference: 3.7993856949469773e-07
	Spike Rate: 0.09576908499002457
-------------
-------------
conditions
	soft_reset: True
	PARM_LEARN: True
	FINISH_STEP: 4
	ACCUMULATE_EVENT_MILITIME: 80
results
	IoU: 0.7514989485343297
	Energy per inference: 7.683567559979565e-07
	Spike Rate: 0.19367557764053345
-------------
-------------
conditions
	soft_reset: True
	PARM_LEARN: True
	FINISH_STEP: 4
	ACCUMULATE_EVENT_MILITIME: 100
results
	IoU: 0.7798413231968879
	Energy per inference: 6.384677817550255e-07
	Spike Rate: 0.1609351634979248
-------------
-------------
conditions
	soft_reset: True
	PARM_LEARN: True
	FINISH_STEP: 8
	ACCUMULATE_EVENT_MILITIME: 10
results
	IoU: 0.49040051677574714
	Energy per inference: 4.748063133774849e-07
	Spike Rate: 0.11968188732862473
-------------
-------------
conditions
	soft_reset: True
	PARM_LEARN: True
	FINISH_STEP: 8
	ACCUMULATE_EVENT_MILITIME: 20
results
	IoU: 0.6811716470619043
	Energy per inference: 4.6717090640413517e-07
	Spike Rate: 0.11775727570056915
-------------
-------------
conditions
	soft_reset: True
	PARM_LEARN: True
	FINISH_STEP: 8
	ACCUMULATE_EVENT_MILITIME: 40
results
	IoU: 0.7707991223533949
	Energy per inference: 7.597654985147528e-07
	Spike Rate: 0.19151002168655396
-------------
-------------
conditions
	soft_reset: True
	PARM_LEARN: True
	FINISH_STEP: 8
	ACCUMULATE_EVENT_MILITIME: 80
results
	IoU: 0.8057607799768448
	Energy per inference: 7.292935038094583e-07
	Spike Rate: 0.18382911384105682
-------------
-------------
conditions
	soft_reset: True
	PARM_LEARN: True
	FINISH_STEP: 8
	ACCUMULATE_EVENT_MILITIME: 100
results
	IoU: 0.820082370142142
	Energy per inference: 1.0379706054663984e-06
	Spike Rate: 0.2616356909275055
-------------

'''

# パターンマッチングを使用して値を抽出
pattern = r'([A-Za-z_ ]+):\s*([0-9.e-]+)'
matches = re.findall(pattern, data)

# 結果をディクショナリに格納
results_dict = {}
for match in matches:
    key = match[0].strip()
    value = match[1].strip()
    results_dict[key] = value

# ディクショナリを表に変換
df = pd.DataFrame(results_dict, index=[0])

print(df.columns)
print(df.head())


# %%
