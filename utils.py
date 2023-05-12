from scipy import integrate, special, interpolate, stats
import numpy as np
import os
# import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from gammapy.irf import load_cta_irfs
from astropy import units as u
import dynesty
irfs = load_cta_irfs('Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits')

edispfull = irfs['edisp']
edispfull.normalize()
bkgfull = irfs['bkg'].to_2d()


edispkernel = edispfull.to_edisp_kernel(offset=1*u.deg)
edispkernel.normalize(axis_name='energy')
log10eaxis = np.log10(edispkernel.axes['energy'].center.value)
log10eaxis = log10eaxis[18:227]
newaxis = np.linspace(log10eaxis[0],log10eaxis[-1],10*(log10eaxis.shape[0]-1)+1)
log10eaxis = newaxis
eaxis = np.power(10., log10eaxis)
eaxis_mod = np.log(eaxis)
logjacob = np.log(np.log(10))+eaxis_mod+np.log(log10eaxis[1]-log10eaxis[0])


def edisp(logerecon,logetrue):
    val = np.log(edispkernel.evaluate(energy_true=np.power(10.,logetrue)*u.TeV, energy = np.power(10.,logerecon)*u.TeV).value)
    norm = special.logsumexp(np.log(edispkernel.evaluate(energy_true=np.power(10.,logetrue)*u.TeV, energy = np.power(10.,log10eaxis)*u.TeV).value)+logjacob)
    return val - norm


# edisp = lambda logerecon, logetrue: stats.norm(loc=10**logetrue, scale=0.5*10**logetrue).logpdf(10**logerecon)



def makedist(centre, spread=0.5, normeaxis=eaxis):
    def distribution(x):
        return stats.norm(loc=np.power(10., centre), scale=spread*np.power(10., centre)).logpdf(np.power(10., x))
    return distribution

def bkgdist(logenerg):
    np.seterr(divide='ignore')
    val  = np.log(bkgfull.evaluate(energy=np.power(10.,logenerg)*u.TeV, offset=1*u.deg).value)
    norm = special.logsumexp(np.log(bkgfull.evaluate(energy=np.power(10.,log10eaxis)*u.TeV, offset=1*u.deg).value)+logjacob)
    np.seterr(divide='warn')
    return val - norm


# def bkgdist(log10eval):
#     return stats.norm(loc=10**-0.5, scale=0.6*np.power(10.,-0.5)).logpdf(10**log10eval)

def logpropdist(logeval):
    func = stats.loguniform(a=10**log10eaxis[0], b=10**log10eaxis[-1])
    return func.logpdf(10**logeval)



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
class COLOR:
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


