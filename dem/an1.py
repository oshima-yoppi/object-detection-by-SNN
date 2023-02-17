import numpy as np


dtype = [('t', '<i4'), ('x', '<i1')]
a = np.zeros((2,2), dtype=dtype)
b = np.ones((2,2))
a['x'] = b
print(a['t'])
print(a)