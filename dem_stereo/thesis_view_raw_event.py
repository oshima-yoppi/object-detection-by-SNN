import h5py
import numpy as np
import matplotlib.pyplot as plt


num = 0
left_raw_event_path = f"raw-data/th-0.15/left/{str(num).zfill(5)}.h5"
right_raw_event_path = f"raw-data/th-0.15/right/{str(num).zfill(5)}.h5"

with h5py.File(left_raw_event_path, "r") as f:
    left_raw_event = f["events"][()]
with h5py.File(right_raw_event_path, "r") as f:
    right_raw_event = f["events"][()]

print(left_raw_event.dtype)
print(left_raw_event[0])

print(left_raw_event[:5, 0])
print(left_raw_event[:5, 1])
print(left_raw_event[:5, 2])
print(left_raw_event[:5, 3])
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(left_raw_event[:, 0], left_raw_event[:, 1], left_raw_event[:, 2], s=0.1)
ax.set_xlabel("time")
ax.set_ylabel("x")
ax.set_zlabel("y")
plt.show()
