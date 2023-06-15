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

offsetaxis = edispfull.axes['offset'].center.value
offsetaxis = offsetaxis[offsetaxis<4]
offsetaxis = np.append(-np.flip(offsetaxis),offsetaxis)


edispkernel = edispfull.to_edisp_kernel(offset=1*u.deg)
edispkernel.normalize(axis_name='energy')
log10eaxis = np.log10(edispkernel.axes['energy'].center.value)


# Restricting energy axis to values that could have non-zero or noisy energy dispersion (psf for energy) values
log10eaxis = log10eaxis[log10eaxis>-0.9]
log10eaxis = log10eaxis[log10eaxis<2.0]


# Usefull mesh values particularly when enforcing normalisation on functions
log10emesh, offsetmesh = np.meshgrid(log10eaxis, offsetaxis)

norme_vals_mesh, normoffset_vals_mesh = np.meshgrid(log10eaxis, offsetaxis)
normtwodcoordinatestacked = np.stack([norme_vals_mesh, normoffset_vals_mesh], axis=-1)

def makelogjacob(log10eaxis=log10eaxis):
    outputlogjacob = np.log(10**log10eaxis)#+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0])
    return outputlogjacob



eaxis = np.power(10., log10eaxis)
eaxis_mod = np.log(eaxis)
logjacob = makelogjacob(log10eaxis)


def edisp(logerecon, logetrue, offsettrue):
    edispfunc = lambda logerecon: np.log(edispfull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                  migra = np.power(10.,logerecon-logetrue), offset=np.abs(offsettrue)*u.deg).value)
    
    normvals = np.array([edispfunc(log10eaxisval) for log10eaxisval in log10eaxis])
    # print(normvals)
    normalisation = special.logsumexp(normvals+logjacob, axis=0)
    
    result = edispfunc(logerecon)-normalisation

    return result

## Testing distribution for the energy dispersion

def psf(offsetrecon, offsettrue, logetrue):
    psffunc = lambda offsetrecon: np.log(psffull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                  rad = (offsettrue-offsetrecon)*u.deg, offset=np.abs(offsettrue)*u.deg).value)
    
    normvals = np.array([psffunc(offsetval)for offsetval in offsetaxis])
    
    normalisation = special.logsumexp(normvals, axis=0)
    
    return psffunc(offsetrecon)-normalisation


# def psf(offsetrecon, offsettrue, logetrue):
#     scale = 0.3*offsettrue
#     psffunc = lambda offsetrecon: -0.5*((offsetrecon-offsettrue)/scale)**2
    
#     normvals = np.array([psffunc(offsetval)for offsetval in offsetaxis])
    
#     normalisation = special.logsumexp(normvals, axis=0)
    
#     return psffunc(offsetrecon)-normalisation

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


def makedist(logmass, spread=0.3, normeaxis=10**log10eaxis):
    eaxis = normeaxis
    def distribution(x):
        log10eaxis = np.log10(eaxis)
        logjacob = makelogjacob(log10eaxis)
        
        specfunc = stats.norm(loc=logmass, scale=spread).logpdf
        
        normfactor = special.logsumexp(specfunc(log10eaxis)+logjacob)
                        
        result = specfunc(x)
        
        return result-normfactor
        
    return distribution


# # Testing distribution for the background

def bkgdist(log10eval, offsetval):
    bkgfunc = stats.multivariate_normal(mean=[-0.5,0], cov=[[0.1,0],[0,2]]).logpdf
    
    log10emesh, offsetmesh = np.meshgrid(log10eval, offsetval)

    twodcoordinates = np.stack([log10emesh, offsetmesh], axis=-1)

    pdfvalues = bkgfunc(twodcoordinates)
    
    bkgnorm = bkgfunc(normtwodcoordinatestacked)
    normfactor = special.logsumexp(bkgnorm.reshape((offsetaxis.shape[0], log10eaxis.shape[0])) + logjacob)
    
    return pdfvalues-normfactor


# Does not have any mention of the log of the jacobian to keep it more general.
def inverse_transform_sampling(logpmf, Nsamples=1):
    
    logpmf = logpmf - special.logsumexp(logpmf)
    logcdf = np.logaddexp.accumulate(logpmf)
    cdf = np.exp(logcdf-logcdf[-1])  

    randvals = [random.random() for xkcd in range(Nsamples)]
    indices = [np.searchsorted(cdf, u) for u in randvals]
    return indices




####################################################################################
####################################################################################
####################################################################################
####################################################################################


# Now including spatial dimensions!


log10emesh, offsetmesh = np.meshgrid(log10eaxis, offsetaxis)


def fake_signal_position_dist(offset):
    nicefunc = stats.multivariate_normal(mean=0, cov=1.0).logpdf
    normfactor = special.logsumexp(nicefunc(offsetaxis))
    return (nicefunc(offset.flatten())-normfactor).reshape(offset.shape)

def setup_full_fake_signal_dist(logmass, normeaxis=10**log10eaxis):
    
    offsetintegrand = special.logsumexp(makedist(logmass, normeaxis=normeaxis)(log10emesh)+fake_signal_position_dist(offsetmesh), axis=0)
    
    normalisation = special.logsumexp(offsetintegrand+makelogjacob(log10eaxis))
    
    def full_fake_signal_dist(log10eval, offsetval):
        pdfvalues = makedist(logmass, normeaxis=normeaxis)(log10eval) + fake_signal_position_dist(offsetval)
        return pdfvalues - normalisation
    
    return full_fake_signal_dist






# def calcirfvals(logemeasured, offsetmeasured, log10eaxis=log10eaxis, offsetaxis=offsetaxis):
#     log10emesh, offsetmesh = np.meshgrid(log10eaxis, offsetaxis)
#     energyloglikelihoodvals=edisp(logemeasured, log10emesh, offsetmesh)
#     pointspreadlikelihoodvals=psf(offsetmeasured, offsetmesh, log10emesh)
    
#     print(np.max(energyloglikelihoodvals))
#     print(np.max(pointspreadlikelihoodvals))
    
#     return energyloglikelihoodvals+pointspreadlikelihoodvals



def calcirfvals(mesauredcoord, log10eaxis=log10eaxis, offsetaxis=offsetaxis):
    logemeasured, offsetmeasured = mesauredcoord
    log10emesh, offsetmesh = np.meshgrid(log10eaxis, offsetaxis)
    energyloglikelihoodvals=edisp(logemeasured, log10emesh, offsetmesh)
    pointspreadlikelihoodvals=psf(offsetmeasured, offsetmesh, log10emesh)
    
    # print(np.max(energyloglikelihoodvals))
    # print(np.max(pointspreadlikelihoodvals))
    
    return energyloglikelihoodvals+pointspreadlikelihoodvals




def evaluateintegral(priorvals, irfvals):
    integrand = priorvals+logjacob+irfvals
    return special.logsumexp(integrand)


def evaluateformass(logmass, irfvals):
    log10emesh, offsetmesh = np.meshgrid(log10eaxis, offsetaxis)

    priorvals = setup_full_fake_signal_dist(logmass, normeaxis=10**log10eaxis)(log10emesh, offsetmesh)
        
    product = np.sum([evaluateintegral(priorvals, irflist) for irflist in irfvals])
    
    return product

