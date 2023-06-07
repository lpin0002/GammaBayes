from scipy import integrate, special, interpolate, stats
import numpy as np
import os
# import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from gammapy.irf import load_cta_irfs
from astropy import units as u
import dynesty


# I believe this is the alpha configuration of the array as there are no LSTs
irfs = load_cta_irfs('Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits')

edispfull = irfs['edisp']
edispfull.normalize()
bkgfull = irfs['bkg'].to_2d()


edispkernel = edispfull.to_edisp_kernel(offset=1*u.deg)
edispkernel.normalize(axis_name='energy')
log10eaxis = np.log10(edispkernel.axes['energy'].center.value)

# Restricting energy axis to values that could have non-zero energy dispersion (psf for energy) values
log10eaxis = log10eaxis[18:227]
# log10eaxis = np.linspace(-1.2,2.2,1800)
## Testing axis of higher resolution than the ones supplied within the IRFs (artificial)
# newaxis = np.linspace(log10eaxis[0],log10eaxis[-1],30*(log10eaxis.shape[0]-1)+1)
# log10eaxis = newaxis
def makelogjacob(log10eaxis=log10eaxis):
    outputlogjacob = np.log(10**log10eaxis)+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0])
    return outputlogjacob

eaxis = np.power(10., log10eaxis)
eaxis_mod = np.log(eaxis)
logjacob = makelogjacob(log10eaxis)


def edisp(logerecon,logetrue):
    probabilityval = np.log(edispkernel.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                 energy = np.power(10.,logerecon)*u.TeV).value)
    normalisationfactor = special.logsumexp(np.log(edispkernel.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                                        energy = np.power(10.,log10eaxis)*u.TeV).value)+logjacob)
    return probabilityval - normalisationfactor

## Testing distribution for the energy dispersion
# edisp = lambda logerecon, logetrue: stats.norm(loc=logetrue, scale=0.1).logpdf(logerecon)



def makedist(logmass, spread=1, normeaxis=eaxis):
    eaxis = normeaxis
    def distribution(x):
        log10eaxis = np.log10(eaxis)
        logjacob = makelogjacob(log10eaxis)
        
        nicefunc = stats.norm(loc=logmass-6, scale=spread).logpdf
        
        normfactor = special.logsumexp(nicefunc(log10eaxis)+logjacob)
        if type(x)==np.ndarray:
            result = np.empty(x.shape)
            
            # This step is using the output of the gaussian for values below the logmass
            #   and then setting all values above to 0 probability essentially
            #   as annihilation shouldn't create particles heavier than the original annihilation pair
            
            np.putmask(result, x<logmass, nicefunc(x[x<logmass]))
            np.putmask(result, x>=logmass, -np.inf)
            # result[x<logmass] = nicefunc(x[x<logmass])
            # result[x>=logmass] = np.full((x[x>=logmass]).shape, -np.inf)
            return result-normfactor
        else:
            if x<logmass:
                return nicefunc(x)-normfactor
            else:
                return -np.inf
    return distribution

def bkgdist(logenerg):
    val  = np.log(bkgfull.evaluate(energy=np.power(10.,logenerg)*u.TeV,
                                   offset=1*u.deg).value)
    norm = special.logsumexp(np.log(bkgfull.evaluate(energy=np.power(10.,log10eaxis)*u.TeV,
                                                     offset=1*u.deg).value)+logjacob)
    return val - norm

# # Testing distribution for the background
# def bkgdist(log10eval):
#     return stats.norm(loc=10**-0.5, scale=0.6*np.power(10.,-0.5)).logpdf(10**log10eval)

def logpropdist(logeval):
    func = stats.loguniform(a=10**log10eaxis[0], b=10**log10eaxis[-1])
    return func.logpdf(10**logeval)


# Does not have any mention of the log of the jacobian to keep it more general.
def inverse_transform_sampling(logpmf, Nsamples=1):
    """Generate a random index using inverse transform sampling.

    Args:
        logpmf: A 1D array of non-negative values representing a log probability mass function.

    Returns:
        A random integer index between 0 and len(pmf) - 1, inclusive.
    """
    logpmf = logpmf - special.logsumexp(logpmf)
    logcdf = np.logaddexp.accumulate(logpmf)
    cdf = np.exp(logcdf-logcdf[-1])  # compute the cumulative distribution function

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


