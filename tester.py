from scipy import integrate, special, interpolate, stats
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from gammapy.irf import load_cta_irfs
from astropy import units as u




irfs = load_cta_irfs('Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits')

edispfull = irfs['edisp']

edispkernel = edispfull.to_edisp_kernel(offset=1*u.deg)
# edisp = lambda erecon, etrue: stats.norm(loc=etrue, scale=(axis[1]-axis[0])).logpdf(erecon)
edisp = lambda erecon, etrue: np.log(edispkernel.evaluate(energy_true=np.power(10.,etrue)*u.TeV, 
                                                   energy = np.power(10.,erecon)*u.TeV).value)
axis = np.log10(edispkernel.axes['energy'].center.value)
axis = axis[18:227]
eaxis = np.power(10., axis)
eaxis_mod = np.log(eaxis)



edispnorms = np.array([special.logsumexp(edisp(axis,axisval)+eaxis_mod) for axisval in axis])


for i, edispval in enumerate(edispnorms):
    print(i, edispval)
