import numpy as np
from scipy import integrate
import os
from scipy import stats, interpolate
import gammapy
from gammapy.irf import EffectiveAreaTable2D, load_cta_irfs
from astropy import units as u
import matplotlib.pyplot as plt
from utils import axis,energydisp, edispkernel

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
    edispsums.append(sum(energydisp(axis,val)))

plt.figure()
plt.plot(edispsums)
plt.show()

