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


offsetbound             = 4.0
trueoffsetsubdivisions  = int(2*offsetbound)
reconoffsetsubdivisions = int(2*offsetbound)
offsetaxis              = np.linspace(-offsetbound,offsetbound, reconoffsetsubdivisions)
offsetaxistrue          = np.linspace(-offsetbound, offsetbound, trueoffsetsubdivisions)

# Restricting energy axis to values that could have non-zero or noisy energy dispersion (psf for energy) values
log10estart             = -1.0
log10eend               = 1.8
log10erange             = log10eend - log10estart
log10eaxis              = np.linspace(log10estart,log10eend,int(np.round(log10erange*5)))
log10eaxistrue          = np.linspace(log10estart,log10eend,int(np.round(log10erange*1000)))


# Usefull mesh values particularly when enforcing normalisation on functions
# log10emesh, offsetmesh = np.meshgrid(log10eaxis, offsetaxis)


def makelogjacob(log10eaxis=log10eaxis):
    outputlogjacob = np.log(10**log10eaxis)#+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0])
    return outputlogjacob

logjacob = makelogjacob(log10eaxis)
logjacobtrue = makelogjacob(log10eaxistrue)


def edisp(logerecon, logetrue, offsettrue):
    
    
    
    return np.log(edispfull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                    migra = np.power(10.,logerecon-logetrue), offset=np.abs(offsettrue)*u.deg).value)


# def edisp(logerecon, logetrue, offsettrue):
#     scale = 1e-2#-1e-3*logetrue)
    
#     return -0.5*((logerecon-logetrue)/scale)**2

## Testing distribution for the energy dispersion

def psf(offsetrecon, offsettrue, logetrue):
    return np.log(psffull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                    rad = (offsettrue-offsetrecon)*u.deg, offset=np.abs(offsettrue)*u.deg).value)


# def psf(offsetrecon, offsettrue, logetrue):
#     scale = 0.1*(offsettrue+1e-3)
    
#     return -0.5*((offsetrecon-offsettrue)/scale)**2

def makedist(logmass, spread=0.3, normeaxis=10**log10eaxis):
    eaxis = normeaxis
    def distribution(x):
        log10eaxis = np.log10(eaxis)
        logjacob = makelogjacob(log10eaxis)
        
        # specfunc = stats.norm(loc=logmass-6, scale=spread).logpdf
        
        specfunc = lambda logenergy: -0.5 * np.log(2 * np.pi * spread**2) - 0.5 * ((logenergy - (logmass-4)) / spread)**2
        
        normfactor = special.logsumexp(specfunc(log10eaxis[log10eaxis<logmass])+logjacob[log10eaxis<logmass])
        
        result = x*0
        
        try:
            belowmassindices = x<logmass
                    
            result[belowmassindices] = specfunc(x[belowmassindices])
            result[x>=logmass] = np.full((x[x>=logmass]).shape, -np.inf)
        except:
            if x<logmass:
                result = specfunc(x)-normfactor
            else:
                result = -np.inf
        
        return result-normfactor
        
    return distribution


# def makedist(logmass, spread=0.1, normeaxis=10**log10eaxis):
#     eaxis = normeaxis
#     def distribution(x):
#         log10eaxis = np.log10(eaxis)
#         logjacob = makelogjacob(log10eaxis)
        
#         specfunc = stats.norm(loc=logmass, scale=spread).logpdf
        
#         normfactor = special.logsumexp(specfunc(log10eaxis)+logjacob)
                        
#         result = specfunc(x)
        
#         return result-normfactor
        
#     return distribution


# # Testing distribution for the background

# def bkgdist(logeval, offsetval):
#     mean = np.array([-0.8, 0.0])
#     cov = np.array([[0.2,0.1],[0.1,1.0]]) 
#     diff = np.column_stack([logeval, offsetval]) - mean
#     exponent = -0.5 * np.sum(diff * np.linalg.solve(cov, diff.T).T, axis=1)
#     log_prob = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(cov)) + exponent
    
#     return log_prob

def bkgdist(logeval, offsetval):
    return np.log(bkgfull.evaluate(energy=10**logeval*u.TeV, offset=np.abs(offsetval)*2*u.deg).value)

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



def setup_full_fake_signal_dist(logmass, specfunc):
    # numpy vectorisation
    def full_fake_signal_dist(log10eval, offsetval):
        log10eval = np.array(log10eval)
        nicespatialfunc = stats.multivariate_normal(mean=0, cov=1.0).logpdf
        if log10eval.ndim>1:
            spectralvals = np.squeeze(specfunc(logmass, log10eval[0,:]))
            spatialvals = np.squeeze(nicespatialfunc(offsetval[:,0]))
            logpdfvalues = spectralvals[np.newaxis,:]+spatialvals[:,np.newaxis]
            return logpdfvalues
        else:
            logpdfvalues = specfunc(logmass, log10eval)+nicespatialfunc(offsetval)
            return logpdfvalues
    
    return full_fake_signal_dist






# def calcirfvals(logemeasured, offsetmeasured, log10eaxis=log10eaxis, offsetaxis=offsetaxis):
#     log10emesh, offsetmesh = np.meshgrid(log10eaxis, offsetaxis)
#     energyloglikelihoodvals=edisp(logemeasured, log10emesh, offsetmesh)
#     pointspreadlikelihoodvals=psf(offsetmeasured, offsetmesh, log10emesh)
    
#     print(np.max(energyloglikelihoodvals))
#     print(np.max(pointspreadlikelihoodvals))
    
#     return energyloglikelihoodvals+pointspreadlikelihoodvals


# edispnormalisations = special.logsumexp(edisp(log10eaxis, log10eaxistrue, offsetaxistrue),axis=1).shape
# psfnormalisations = special.logsumexp(psf(offsetaxis, offsetaxistrue, log10eaxistrue),axis=1).shape

offsettruemeshpsf, offsetreconmeshpsf, logetruemeshpsf = np.meshgrid(offsetaxistrue, offsetaxis, log10eaxistrue)
psfnormalisations = special.logsumexp(psf(offsetreconmeshpsf.flatten(), offsettruemeshpsf.flatten(), logetruemeshpsf.flatten()).reshape(offsetreconmeshpsf.shape),axis=0)

logetruemeshedisp, logereconmeshedisp, offsettruemeshedisp,  = np.meshgrid(log10eaxistrue, log10eaxis, offsetaxistrue)
edispnormalisations = special.logsumexp(edisp(logereconmeshedisp.flatten(), logetruemeshedisp.flatten(), offsettruemeshedisp.flatten()).reshape(logereconmeshedisp.shape).T +logjacob,axis=2)


def calcirfvals(measuredcoord, log10eaxis=log10eaxis, offsetaxis=offsetaxis):
    logemeasured, offsetmeasured    = measuredcoord
    log10emesh, offsetmesh          = np.meshgrid(log10eaxistrue, offsetaxistrue)
    energyloglikelihoodvals         = edisp(logemeasured, log10emesh, offsetmesh)-edispnormalisations
    pointspreadlikelihoodvals       = psf(offsetmeasured, offsetmesh, log10emesh)-psfnormalisations
    
    
    return energyloglikelihoodvals+pointspreadlikelihoodvals




def evaluateintegral(irfvals, priorvals):
    integrand = priorvals+logjacobtrue+irfvals
    return special.logsumexp(integrand)


log10emeshtrue, offsetmeshtrue = np.meshgrid(log10eaxistrue, offsetaxistrue)

def evaluateformass(logmass, irfvals, specfunc):
    priorfunc = setup_full_fake_signal_dist(logmass, specfunc=specfunc)
    
    
    sigpriorvalues = np.squeeze(priorfunc(log10emeshtrue, offsetmeshtrue))
    
    normalisation = special.logsumexp(sigpriorvalues+logjacobtrue)

        
    signalmarginalisationvalues = [evaluateintegral(priorvals=sigpriorvalues-normalisation, irfvals=irflist) for irflist in irfvals]
    
    return signalmarginalisationvalues



import matplotlib.transforms as transforms
import matplotlib.patches as patches

def confidence_ellipse(x, y, probabilities, ax, n_std=3.0, edgecolor='black',facecolor='none', **kwargs):
    
    if not isinstance(probabilities, np.ndarray):
        probabilities = np.array(probabilities)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2-dimensional array")

    n, m = probabilities.shape

    
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    cov = np.cov(x, y, aweights=probabilities.flatten())
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = patches.Ellipse((0, 0),
                              width=ell_radius_x * 2,
                              height=ell_radius_y * 2,
                              edgecolor = edgecolor,
                              facecolor = facecolor,
                              **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.sum(x * probabilities.flatten()) / np.sum(probabilities)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.sum(y * probabilities.flatten()) / np.sum(probabilities)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)