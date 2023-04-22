from scipy import integrate, special, interpolate, stats
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from gammapy.irf import load_cta_irfs
from astropy import units as u
import dynesty
irfs = load_cta_irfs('Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits')

edispfull = irfs['edisp']
bkgfull = irfs['bkg'].to_2d()


edispkernel = edispfull.to_edisp_kernel(offset=1*u.deg)
axis = np.log10(edispkernel.axes['energy'].center.value)
axis = axis[18:227]
eaxis = np.power(10., axis)
eaxis_mod = np.log(eaxis)
logjacob = np.log(np.log(10))+eaxis_mod+np.log(axis[1]-axis[0])


# def edisp(logerecon,logetrue):
#     val = np.log(edispkernel.evaluate(energy_true=np.power(10.,logetrue)*u.TeV, energy = np.power(10.,logerecon)*u.TeV).value)
#     norm = special.logsumexp(eaxis_mod+np.log(edispkernel.evaluate(energy_true=np.power(10.,logetrue)*u.TeV, energy = np.power(10.,axis)*u.TeV).value))
#     return val-norm


edisp = lambda erecon, etrue: stats.norm(loc=etrue, scale=(axis[1]-axis[0])).logpdf(erecon)



def makedist(centre, spread=0.3):
    func = lambda x: stats.norm(loc=np.power(10., centre), scale=spread*np.power(10.,centre)).logpdf(np.power(10., x))
    return func


bkgdist = lambda logenerg: np.log(bkgfull.evaluate(energy=np.power(10.,logenerg)*u.TeV, offset=1*u.deg).value)

# bkgdist = makedist(-0.5)



def inverse_transform_sampling(logpmf, Nsamples=1):
    """Generate a random index using inverse transform sampling.

    Args:
        logpmf: A 1D array of non-negative values representing a log probability mass function.

    Returns:
        A random integer index between 0 and len(pmf) - 1, inclusive.
    """
    logpmf = logpmf - special.logsumexp(logpmf)
    pmf = np.exp(logpmf)
    cdf = np.cumsum(pmf)  # compute the cumulative distribution function
    # print(cdf)
    randvals = [random.random() for xkcd in range(Nsamples)]
    indices = [np.searchsorted(cdf, u) for u in randvals]
    return indices



# Purely for different looking terminal outputs
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

