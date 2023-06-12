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
psffull = irfs['psf']
edispfull.normalize()
bkgfull = irfs['bkg'].to_2d()

offsetaxis = psffull.axes['offset'].center.value
offsetaxis = np.append(-np.flip(offsetaxis),offsetaxis)

# offsetaxis = np.linspace(-5.5,5.5,120)



edispkernel = edispfull.to_edisp_kernel(offset=1*u.deg)
edispkernel.normalize(axis_name='energy')
log10eaxis = np.log10(edispkernel.axes['energy'].center.value)


# Restricting energy axis to values that could have non-zero energy dispersion (psf for energy) values
log10eaxis = log10eaxis[18:227]

log10emesh, offsetmesh = np.meshgrid(log10eaxis, offsetaxis)

norme_vals_mesh, normoffset_vals_mesh = np.meshgrid(log10eaxis, offsetaxis)
normtwodcoordinatestacked = np.stack([norme_vals_mesh, normoffset_vals_mesh], axis=-1)

def makelogjacob(log10eaxis=log10eaxis):
    outputlogjacob = np.log(10**log10eaxis)#+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0])
    return outputlogjacob



eaxis = np.power(10., log10eaxis)
eaxis_mod = np.log(eaxis)
logjacob = makelogjacob(log10eaxis)


# def edisp(logerecon,logetrue):
#     probabilityval = np.log(edispkernel.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
#                                                  energy = np.power(10.,logerecon)*u.TeV).value)
#     normalisationfactor = special.logsumexp(np.log(edispkernel.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
#                                                                         energy = np.power(10.,log10eaxis)*u.TeV).value)+logjacob)
#     return probabilityval - normalisationfactor

## Testing distribution for the energy dispersion

#norme_vals_mesh, normoffset_vals_mesh = np.meshgrid(log10eaxis, offsetaxis)

def edisp(logerecon, logetrue, offsettrue):
    scale = 10**(logetrue-1) 
    edispfunc = lambda logerecon: -0.5*((10**logerecon-10**logetrue)**2/scale**2)
    
    normvals = np.array([edispfunc(log10eaxisval) for log10eaxisval in log10eaxis])

    normalisation = special.logsumexp(normvals+logjacob, axis=0)
    # normalisation = 0.0
    
    result = edispfunc(logerecon)-normalisation

    return result

def psf(offsetrecon, offsettrue, logetrue):
    scale = 0.1*offsettrue
    psffunc = lambda offsetrecon: -0.5*((offsetrecon-offsettrue)/scale)**2
    
    normvals = np.array([psffunc(offsetval)for offsetval in offsetaxis])
    
    normalisation = special.logsumexp(normvals, axis=0)
    
    return psffunc(offsetrecon)-normalisation

# def makedist(logmass, spread=1, normeaxis=10**log10eaxis):
#     eaxis = normeaxis
#     def distribution(x):
#         log10eaxis = np.log10(eaxis)
#         logjacob = makelogjacob(log10eaxis)
        
#         specfunc = stats.norm(loc=logmass-6, scale=spread).logpdf
        
#         normfactor = special.logsumexp(specfunc(log10eaxis)+logjacob)
        
#         result = x*0
                
#         result[x<logmass] = specfunc(x[x<logmass])
#         result[x>=logmass] = np.full((x[x>=logmass]).shape, -np.inf)
        
#         return result-normfactor
        
#     return distribution


def makedist(logmass, spread=0.4, normeaxis=10**log10eaxis):
    eaxis = normeaxis
    def distribution(x):
        log10eaxis = np.log10(eaxis)
        logjacob = makelogjacob(log10eaxis)
        
        specfunc = stats.norm(loc=logmass, scale=spread).logpdf
        
        normfactor = special.logsumexp(specfunc(log10eaxis)+logjacob)
                        
        result = specfunc(x)
        
        return result-normfactor
        
    return distribution

# def bkgdist(logenerg):
#     val  = np.log(bkgfull.evaluate(energy=np.power(10.,logenerg)*u.TeV,
#                                    offset=1*u.deg).value)
#     norm = special.logsumexp(np.log(bkgfull.evaluate(energy=np.power(10.,log10eaxis)*u.TeV,
#                                                      offset=1*u.deg).value)+logjacob)
#     return val - norm

# # Testing distribution for the background
# def bkgdist(log10eval):
#     nicefunc = stats.norm(loc=10**-0.5, scale=0.6*np.power(10.,-0.5)).logpdf
#     normfactor = special.logsumexp(nicefunc(log10eaxis)+logjacob)
#     return nicefunc(10**log10eval)-normfactor



def bkgdist(log10eval, offsetval):
    bkgfunc = stats.multivariate_normal(mean=[-0.5,0], cov=[[0.1,0],[0,2]]).logpdf
    
    log10emesh, offsetmesh = np.meshgrid(log10eval, offsetval)

    twodcoordinates = np.stack([log10emesh, offsetmesh], axis=-1)

    pdfvalues = bkgfunc(twodcoordinates)
    
    bkgnorm = bkgfunc(normtwodcoordinatestacked)
    normfactor = special.logsumexp(bkgnorm.reshape((offsetaxis.shape[0], log10eaxis.shape[0])) + logjacob)
    
    return pdfvalues-normfactor

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



####################################################################################
####################################################################################
####################################################################################
####################################################################################


# Now including spatial dimensions!


def fake_signal_position_dist(offset):
    nicefunc = stats.multivariate_normal(mean=0, cov=1.0).logpdf
    normfactor = special.logsumexp(nicefunc(offsetaxis))
    return (nicefunc(offset.flatten())-normfactor).reshape(offset.shape)

def setup_full_fake_signal_dist(logmass, normeaxis=10**log10eaxis):
    offsetintegrand = special.logsumexp(makedist(logmass, normeaxis=normeaxis)(norme_vals_mesh)+fake_signal_position_dist(normoffset_vals_mesh), axis=0)
    
    normalisation = special.logsumexp(offsetintegrand+makelogjacob(log10eaxis))
    
    def full_fake_signal_dist(log10eval, offsetval):
        pdfvalues = makedist(logmass, normeaxis=normeaxis)(log10eval) + fake_signal_position_dist(offsetval)
        return pdfvalues - normalisation
    
    return full_fake_signal_dist




def evaluateintegral(priorvals, logemeasured, offsetmeasured, log10emesh, offsetmesh):
    
    energyloglikelihoodvals=edisp(logemeasured, log10emesh, offsetmesh)
    pointspreadlikelihoodvals=psf(offsetmeasured, offsetmesh, log10emesh)
    integrand = priorvals+logjacob+energyloglikelihoodvals+pointspreadlikelihoodvals
    return special.logsumexp(integrand)


def evaluateformass(logmass, logemasuredvals, offsetmeasuredvals):
    log10emesh, offsetmesh = np.meshgrid(log10eaxis, offsetaxis)

    priorvals = setup_full_fake_signal_dist(logmass, normeaxis=10**log10eaxis)(log10emesh, offsetmesh)
        
    product = np.sum([evaluateintegral(priorvals, logemeasured, offsetmeasured, log10emesh, offsetmesh) for logemeasured, offsetmeasured in zip(logemasuredvals, offsetmeasuredvals)])
    
    return product

