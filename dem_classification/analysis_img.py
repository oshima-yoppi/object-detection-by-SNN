import os
import glob
from collections import defaultdict

th_keys = ["0.05", "0.1", "0.15"]
all_fn_number = set()
common_fn_number = []
false_numer = defaultdict(list)
for th in th_keys:
    all_path = glob.glob(
        f"result_img/Conv3Full2_80000_(130,173)_th-{th}_para-False_TimeChange-False_step-(5,5)_Reset-subtract_EventCount-False_Distinguish-True_LeargeData-True/FN_images/*"
    )
    print(len(all_path))
    for path in all_path:

        number = int(os.path.basename(path).split("_")[0][:-4])
        if number in all_fn_number:
            common_fn_number.append(number)
        all_fn_number.add(number)
        false_numer[th].append(number)


for key in th_keys:
    for number in false_numer[key]:
        if number in common_fn_number:
            false_numer[key].remove(number)
# a = common_fn_number
# print(a)
print(common_fn_number)
print(false_numer)
for key in th_keys:
    print(len(false_numer[key]))
