import numpy as np

x = np.array([1,2,3,4])
print(x.shape[0])

print(np.repeat(x,3).reshape((x.shape[0],3)).T)


