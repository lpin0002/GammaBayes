import numpy as np
from scipy import integrate
import os
from scipy import stats, interpolate, sparse
import gammapy
from gammapy.irf import EffectiveAreaTable2D, load_cta_irfs
from astropy import units as u
import matplotlib.pyplot as plt
from utils import axis,energydisp, edispkernel
from sys import getsizeof


print(axis)
# edispkernel.normalize('energy')

plt.figure()
edispplot = edispkernel.plot_matrix(add_cbar=True)
plt.show()

def energydisp(log_energy_measured, log_energy_true):
     return edispkernel.evaluate(energy_true=np.power(10.,log_energy_true)*u.TeV, energy = np.power(10.,log_energy_measured)*u.TeV).value

edispintegrals = []

for val in axis:
    edispintegrals.append(integrate.simps(y=energydisp(val,axis), x=np.power(10.,axis)))

plt.figure()
plt.plot(edispintegrals)
plt.show()

edispsums = []

for val in axis:
    edispsums.append(np.sum(energydisp(axis,val)))

plt.figure()
plt.plot(edispsums)
plt.show()



arr = np.full((1000, 1000), np.nan)

# Create a masked array from the original array
masked_arr = np.ma.masked_invalid(arr)


# Create a sparse matrix that treats NaN values as empty values
n_rows = 2
n_cols = 2
sparse_matrix = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
normal_matrix = [[],[]]

# Set some values
sparse_matrix[0, 0] = 1.0
sparse_matrix[1, 1] = 2.0
sparse_matrix[0, 1] = -np.nan
sparse_matrix[1,0] = -np.nan

indices = np.nonzero(~np.isnan(sparse_matrix))
sps = sparse.coo_matrix((sparse_matrix[indices],indices), shape=sparse_matrix.shape)
# Set some values
normal_matrix = [[1.0,-np.nan],[-np.nan,2.0]]


sps.toarray()
# Access some values
print("sparse_matrix[0, 0] =", sparse_matrix[0][0])
print("sparse_matrix[1, 1] =", sparse_matrix[1][1])
print("sparse_matrix[0,1] =", sparse_matrix[0][1])


# Access some values
print("normal_matrix[0, 0] =", normal_matrix[0][0])
print("normal_matrix[1, 1] =", normal_matrix[1][1])
print("normal_matrix[0,1] =", normal_matrix[0][1])

print("Sum of normal", np.nansum(normal_matrix, axis=1))
print("Sum of sparse", np.sum(sparse_matrix, axis=1))

print("Size of normal", getsizeof(normal_matrix))
print("Size of sparse", getsizeof(sparse_matrix))