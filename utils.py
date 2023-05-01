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



def makedist(centre, spread=0.5):
    func = lambda x: stats.norm(loc=np.power(10., centre), scale=spread*np.power(10.,centre)).logpdf(np.power(10., x))
    return func


def makedist(centre, spread=0.5):
    energycentre = 10**centre
    espread = spread*energycentre
    func = lambda logenergy: np.log((2*np.pi)**(-0.5)*espread**-1)-(10**logenergy-energycentre)**2/(2*espread**2)
    return func

# def bkgdist(logenerg):
#     return np.log(bkgfull.evaluate(energy=np.power(10.,logenerg)*u.TeV, offset=1*u.deg).value)


def bkgdist(log10eval):
    return stats.norm(loc=10**-0.5, scale=0.6*np.power(10.,-0.5)).logpdf(10**log10eval)

def logpropdist(logeval):
    func = stats.uniform(loc=10**axis[0], scale=10**axis[-1]-10**axis[0])
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


def printimportant(numevents, truelogmass, truelambda, axis=axis, logmassrange=None, lambdarange=None):

    stringtoprint ="\n\n"

    stringtoprint +=f"""IMPORTANT PARAMETERS:"""
    stringtoprint +=f"""number of events  being analysed/were simulated is {numevents:.1e}."""

    stringtoprint +=f"""true log mass value  used for the signal model is {truelogmass} or equivalently a mass of roughly {np.round(np.power(10., truelogmass),3):.2e}."""

    stringtoprint +=f"""fraction of signal events to total events is {truelambda}."""

    stringtoprint +=f"""bounds for the log energy range  are {axis[0]:.2e} and {axis[-1]:.2e} translating into energy bounds of {np.power(10.,axis[0]):.2e} and {np.power(10.,axis[-1]):.2e}."""

    if not(logmassrange ==None): 
        stringtoprint +=f"""bounds for the log mass range [TeV]  are {logmassrange[0]:.2e} and {logmassrange[-1]:.2e} translating into mass bounds of {np.power(10.,logmassrange[0]):.2e} and {np.power(10.,logmassrange[-1]):.2e} [TeV]."""

    
    if not(lambdarange==None):
        stringtoprint +=f"""bounds for the lambda range are {lambdarange[0]:.2e} and {lambdarange[-1]:.2e}."""

    stringtoprint+="\n"

    print(stringtoprint)
