#%%
import tonic
import h5py
import numpy as np
import tonic.transforms as transforms
import time
#%%

# dataset = tonic.datasets.NMNIST(save_to='data_', train=True)


# dataset = tonic.datasets.NMNIST(save_to='data_', train=False)
# events_, target = dataset[0]
# print(type(events_.shape))
# print(type(events_))
# tonic.utils.plot_event_grid(events_)
# %%

h5py_path = f"data/0.h5"
youtube_path = f"0.gif"
with h5py.File(h5py_path, "r") as f:
    label = f['label'][()]
    events = f['events'][()]
dtype = [('t', '<i4'), ('x', '<i4'), ('y', '<i4'), ('p', '<i4')]
print(label)
print(events)
print(events.shape, events.dtype)
start = time.time()
event_len = events.shape[0]
events_change = np.zeros(event_len, dtype=dtype)
for i, (key , _) in enumerate(dtype):
    events_change[key] = events[:,i]
    print(max(events[:,i]))
print(events_change, events_change.shape)
print(time.time()-start)
transform = transforms.ToFrame(sensor_size=(361, 360, 2), time_window= 100000)
# transform = transforms.ToFrame(time_window= 1000)
new = transform(events_change)
new.shape
# tonic.utils.plot_event_grid(events_change, axis_array=(2,5))
# %%
# time =  events['t']
# time.shape
# %%
dt = [('x', 'i2'), ('y', 'i2')]
data = np.array([[1,2], [3,4]])
data = data.astype(dt)
data =data.tolist()
# data_ = data.astype(dt)
type(data)
data
# %%
