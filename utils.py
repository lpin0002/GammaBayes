import os
from scipy import stats, interpolate
import numpy as np
import gammapy
from gammapy.irf import EffectiveAreaTable2D, load_cta_irfs
from astropy import units as u
irfs = load_cta_irfs("Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits")


edispkernel =irfs['edisp'].to_edisp_kernel(offset=1*u.deg)

# def energydisp(log_energy_measured, log_energy_true):
#      return edispkernel.evaluate(energy_true=np.power(10.,log_energy_true)*u.TeV, energy = np.power(10.,log_energy_measured)*u.TeV)

axis = np.log10(edispkernel.axes["energy_true"].center.value)
axis = axis[18:227]
energydisp = lambda log_energy_measured, log_energy_true: stats.norm(loc=log_energy_true, scale = 1e-3*(3-log_energy_true)+1e-3).pdf(log_energy_measured)

# axis = np.linspace(-1.5,2.5,300)

edispnorms = []
for val in axis:
     edispnorms.append(np.sum(energydisp(axis, val)))
edispnorms = np.array(edispnorms)

def find_closest(arr, val):
       idx = np.abs(arr - val).argmin()
       return arr[idx]



def make_gaussian(centre = 0.25, spread = 0.1, axis=axis):
    continuous_gaussian = stats.norm(loc=centre, scale=spread)

    norm = np.sum(continuous_gaussian.pdf(axis))
    outputfunc = lambda x: continuous_gaussian.pdf(x)/norm

    return outputfunc


backgrounddist = make_gaussian(centre = find_closest(axis,0.25), axis=axis)





def inv_trans_sample(Nsamples, pdf):

    pdf = pdf/np.sum(pdf)
    uniformvals = np.random.uniform(size=Nsamples)

    cdf = np.cumsum(pdf)

    indices= np.array(range(len(axis)))

    interpolationfunc = interpolate.interp1d(y = indices, x=cdf, kind='nearest', fill_value=(0, indices[-1]), bounds_error=False)

    sampledindices = np.int64(interpolationfunc(uniformvals))

    return sampledindices

