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
offsetaxis = np.arange(-offsetaxis[-1],offsetaxis[-1]+0.02, step=0.02)


edispkernel = edispfull.to_edisp_kernel(offset=1*u.deg)
edispkernel.normalize(axis_name='energy')
log10eaxis = np.log10(edispkernel.axes['energy'].center.value)


# Restricting energy axis to values that could have non-zero or noisy energy dispersion (psf for energy) values
log10eaxis = log10eaxis[log10eaxis>-0.8]
log10eaxis = log10eaxis[log10eaxis<1.8]


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
    
    normvals = np.array([edispfunc(log10eval) for log10eval in log10eaxis])
    # print(normvals)
    normalisation = special.logsumexp(normvals.T+logjacob, axis=0)
    
    
    result = edispfunc(logerecon)-normalisation

    return result


# def edisp(logerecon, logetrue, offsettrue):
#     scale = 1e-2#-1e-3*logetrue)
#     edispfunc = lambda logerecon: -0.5*((logerecon-logetrue)/scale)**2
    
#     normvals = np.array([edispfunc(logerecon) for logerecon in log10eaxis])
    
#     normalisation = special.logsumexp(normvals+logjacob, axis=0)
    
#     return edispfunc(logerecon)-normalisation

## Testing distribution for the energy dispersion

def psf(offsetrecon, offsettrue, logetrue):
    psffunc = lambda offsetrecon: np.log(psffull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                  rad = (offsettrue-offsetrecon)*u.deg, offset=np.abs(offsettrue)*u.deg).value)
    
    normvals = [psffunc(offsetval) for offsetval in offsetaxis]
    
    normalisation = special.logsumexp(normvals, axis=0)
    
    return psffunc(offsetrecon)-normalisation


# def psf(offsetrecon, offsettrue, logetrue):
#     scale = 0.1*(offsettrue+1e-3)
#     psffunc = lambda offsetrecon: -0.5*((offsetrecon-offsettrue)/scale)**2
    
#     normvals = np.array([psffunc(offsetval)for offsetval in offsetaxis])
    
#     normalisation = special.logsumexp(normvals, axis=0)
    
#     return psffunc(offsetrecon)-normalisation

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
    return np.log(bkgfull.evaluate(energy=10**logeval*u.TeV, offset=np.abs(offsetval)*u.deg).value)

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
    return nicefunc(offset)-normfactor

def setup_full_fake_signal_dist(logmass, specsetup, normeaxis=10**log10eaxis):
    
    sigpriorvalues = []

    for ii, logeval in enumerate(log10eaxis):
        singlerow = specsetup(logmass,normeaxis=10**log10eaxis)(logeval) + fake_signal_position_dist(offsetaxis)
        sigpriorvalues.append(singlerow)
    sigpriorvalues = np.array(sigpriorvalues).T
    
    normalisation = special.logsumexp(sigpriorvalues+makelogjacob(log10eaxis))
    
    # normalisation = 0.0
    
    # TODO: Make namespace more readable to someone unfamiliar with code
    # numpy vectorisation
    def full_fake_signal_dist(log10eval, offsetval):
        logpdfvalues = specsetup(logmass, normeaxis=normeaxis)(log10eval) + fake_signal_position_dist(offsetval)
        return logpdfvalues - normalisation
    
    return full_fake_signal_dist






# def calcirfvals(logemeasured, offsetmeasured, log10eaxis=log10eaxis, offsetaxis=offsetaxis):
#     log10emesh, offsetmesh = np.meshgrid(log10eaxis, offsetaxis)
#     energyloglikelihoodvals=edisp(logemeasured, log10emesh, offsetmesh)
#     pointspreadlikelihoodvals=psf(offsetmeasured, offsetmesh, log10emesh)
    
#     print(np.max(energyloglikelihoodvals))
#     print(np.max(pointspreadlikelihoodvals))
    
#     return energyloglikelihoodvals+pointspreadlikelihoodvals



def calcirfvals(measuredcoord, log10eaxis=log10eaxis, offsetaxis=offsetaxis):
    logemeasured, offsetmeasured = measuredcoord
    log10emesh, offsetmesh = np.meshgrid(log10eaxis, offsetaxis)
    energyloglikelihoodvals=edisp(logemeasured, log10emesh, offsetmesh)
    pointspreadlikelihoodvals=psf(offsetmeasured, offsetmesh, log10emesh)
    
    # print(np.max(energyloglikelihoodvals))
    # print(np.max(pointspreadlikelihoodvals))
    
    return energyloglikelihoodvals+pointspreadlikelihoodvals




def evaluateintegral(irfvals, priorvals):
    integrand = priorvals+logjacob+irfvals
    return special.logsumexp(integrand)


def evaluateformass(logmass, irfvals, specsetup):
    priorfunc = setup_full_fake_signal_dist(logmass, specsetup=specsetup, normeaxis=10**log10eaxis)
    
    priorvals = []
    for offsetval in offsetaxis:
        priorvals.append(priorfunc(log10eaxis, offsetval))
    priorvals = np.array(priorvals)
    
    normalisation = special.logsumexp(priorvals+logjacob)

        
    signalmarginalisationvalues = [evaluateintegral(priorvals=priorvals-normalisation, irfvals=irflist) for irflist in irfvals]
    
    return signalmarginalisationvalues



import matplotlib.transforms as transforms
import matplotlib.patches as patches

def confidence_ellipse(x, y, probabilities, ax, n_std=3.0, edgecolor='white',facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse based on a grid of probabilities.

    Parameters
    ----------
    probabilities : array-like, shape (n, m)
        Grid of probability values.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
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