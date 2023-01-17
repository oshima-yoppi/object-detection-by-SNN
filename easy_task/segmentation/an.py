#%%
# import torch
# import torch.nn as nn
# c = nn.MSELoss()
# a = torch.zeros((2,3))
# b = torch.ones((2,3))
# print(b * 5)
a = [i for i in range(5)]
a

# %%
try:
    a = b
except Exception as e:
    import traceback
    print('--------error--------')
    traceback.print_exc()
    print('--------error--------')
print(99999)