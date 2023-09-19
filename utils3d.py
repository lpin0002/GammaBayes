from scipy import integrate, special, interpolate, stats
import numpy as np
import os
# import matplotlib.pyplot as plt
import random, time
from tqdm import tqdm
from gammapy.irf import load_cta_irfs
from astropy import units as u
import dynesty
import gc
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'GammaBayes'))

from gammapy.astro.darkmatter import (
    profiles,
    JFactory
)

profile = profiles.EinastoProfile()

# Adopt standard values used in HESS
profiles.DMProfile.DISTANCE_GC = 8.5 * u.kpc
profiles.DMProfile.LOCAL_DENSITY = 0.39 * u.Unit("GeV / cm3")

profile.scale_to_local_density()


np.seterr(divide = 'ignore')
# I believe this is the alpha configuration of the array as there are no LSTs
irfs = load_cta_irfs('Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits')




def convertlonlat_to_offset(fov_coord):
    # Currently assuming small angles (|angle|<=4)
    return np.linalg.norm(fov_coord, axis=0)


def angularseparation(coord1, coord2=None):
    # Currently assuming small angles (|angle|<=4)
    
    try:
        return np.linalg.norm(coord2-coord1, axis=0)
    except:
        try:
            return np.linalg.norm(coord2-coord1.T, axis=1)
        except:
            return np.linalg.norm(coord2.T-coord1, axis=1)

def angularseparation_quick(coord1, coord2=None):
    # Currently assuming small angles (|angle|<=4)
    
    return np.linalg.norm(coord2-coord1, axis=0)



edispfull = irfs['edisp']
psffull = irfs['psf']
edispfull.normalize()
bkgfull = irfs['bkg']
psf3d = psffull.to_psf3d()
aefffull = irfs['aeff']

offsetaxis = psf3d.axes['rad'].center.value

bkgfull2d = bkgfull.to_2d()
bkgfull2doffsetaxis = bkgfull2d.axes['offset'].center.value
offsetaxisresolution = bkgfull2doffsetaxis[1]-bkgfull2doffsetaxis[0] # Comes out to 0.2
latbound            = 3.
lonbound            = 3.5



latitudeaxis            = np.linspace(-latbound, latbound, int(round(2*latbound/0.4)))
latitudeaxistrue        = np.linspace(-latbound, latbound, int(round(2*latbound/0.2)))

longitudeaxis           = np.linspace(-lonbound, lonbound, int(round(2*lonbound/0.4))) 
longitudeaxistrue       = np.linspace(-lonbound, lonbound, int(round(2*lonbound/0.2))) 


# Restricting energy axis to values that could have non-zero or noisy energy dispersion (psf for energy) values
log10estart             = -1.0
log10eend               = 2.4
log10erange             = log10eend - log10estart
log10eaxis              = np.linspace(log10estart,log10eend,int(np.round(log10erange*50))+1)
log10eaxistrue          = np.linspace(log10estart,log10eend,int(np.round(log10erange*100))+1)


# Axes used to plotting the discrete values
plotlongitudeaxis = np.append(longitudeaxis-0.5*np.abs(longitudeaxis[1]-longitudeaxis[0]), longitudeaxis[-1]+0.5*np.abs(longitudeaxis[1]-longitudeaxis[0]))
plotlongitudeaxistrue = np.append(longitudeaxistrue-0.5*np.abs(longitudeaxistrue[1]-longitudeaxistrue[0]), longitudeaxistrue[-1]+0.5*np.abs(longitudeaxistrue[1]-longitudeaxistrue[0]))
plotlatitudeaxis = np.append(latitudeaxis-0.5*np.abs(latitudeaxis[1]-latitudeaxis[0]), latitudeaxis[-1]+0.5*np.abs(latitudeaxis[1]-latitudeaxis[0]))
plotlatitudeaxistrue = np.append(latitudeaxistrue-0.5*np.abs(latitudeaxistrue[1]-latitudeaxistrue[0]), latitudeaxistrue[-1]+0.5*np.abs(latitudeaxistrue[1]-latitudeaxistrue[0]))

plotlog10eaxis = np.append(log10eaxis-0.5*np.abs(log10eaxis[1]-log10eaxis[0]), log10eaxis[-1]+0.5*np.abs(log10eaxis[1]-log10eaxis[0]))
plotlog10eaxistrue = np.append(log10eaxistrue-0.5*np.abs(log10eaxistrue[1]-log10eaxistrue[0]), log10eaxistrue[-1]+0.5*np.abs(log10eaxistrue[1]-log10eaxistrue[0]))


def makelogjacob(log10eaxis=log10eaxis):
    outputlogjacob = np.log(10**log10eaxis)#+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0])
    return outputlogjacob

logjacob = makelogjacob(log10eaxis)
logjacobtrue = makelogjacob(log10eaxistrue)


def edisp(logereconstructed, logetrue, truespatialcoord):
    return np.log(edispfull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                    migra = np.power(10.,logereconstructed-logetrue), 
                                                    offset=convertlonlat_to_offset(truespatialcoord)*u.deg).value)
    
    
def edisp_test(logereconstructed, logetrue, lonval, latval):
    return np.log(edispfull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                    migra = np.power(10.,logereconstructed-logetrue), 
                                                    offset=convertlonlat_to_offset(np.array([lonval, latval]))*u.deg).value)

def edisp_efficient(logereconstructed, logetrue, offset):
    return np.log(edispfull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                    migra = np.power(10.,logereconstructed-logetrue), 
                                                    offset=offset*u.deg).value)
def aeff_efficient(logetrue, offset):
    return aefffull.evaluate(energy_true=10**logetrue*u.TeV, offset=offset*u.deg).to(u.cm**2)

def psf(reconstructed_spatialcoord, logetrue, truespatialcoord):
    
    rad = angularseparation(reconstructed_spatialcoord, truespatialcoord).flatten()
    offset  = convertlonlat_to_offset(truespatialcoord).flatten()
    energyvals = np.power(10.,logetrue.flatten())
    output = np.log(psffull.evaluate(energy_true=energyvals*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    return output


def psf_test(recon_lon, recon_lat, logetrue, true_lon, true_lat):
    reconstructed_spatialcoord = np.array([recon_lon, recon_lat])
    truespatialcoord = np.array([true_lon, true_lat])
    rad = angularseparation(reconstructed_spatialcoord, truespatialcoord).flatten()
    offset  = convertlonlat_to_offset(truespatialcoord).flatten()
    energyvals = np.power(10.,logetrue)
    output = np.log(psffull.evaluate(energy_true=energyvals*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    return output

def psf_efficient(rad, logetrue, offset):

    output = np.log(psffull.evaluate(energy_true=10**logetrue*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    return output



def log_squared_einasto_lb(spatialcoords):
    # fov_longitude, fov_latitude = spatialcoords
    # Using the einasto profile from the paper https://iopscience.iop.org/article/10.1088/1475-7516/2021/01/057/pdf
    # alpha = 0.17, r_s = 20 kpc, rho_s = 0.081 GeV/cm^3
    # distance from galactic centre ~ 8.5 kpc
    angularoffset = convertlonlat_to_offset(spatialcoords.T)
    r = 8.5 * angularoffset
    density = 0.081*np.exp(-(2/0.17)*((r/20)**0.17-1))
    return np.log(density**2)

def makedist(logmass, spread=0.3, normeaxis=10**log10eaxis):
    eaxis = normeaxis
    def distribution(x):
        log10eaxis = np.log10(eaxis)
        logjacob = makelogjacob(log10eaxis)
        
        # specfunc = stats.norm(loc=logmass-6, scale=spread).logpdf
        
        specfunc = lambda logenergy: -0.5 * np.log(2 * np.pi * spread**2) - 0.5 * ((logenergy - (logmass-4)) / spread)**2
        
        result = x*0
        
        try:
            belowmassindices = x<logmass
                    
            result[belowmassindices] = specfunc(x[belowmassindices])
            result[x>=logmass] = np.full((x[x>=logmass]).shape, -np.inf)
        except:
            if x<logmass:
                result = specfunc(x)
            else:
                result = -np.inf
        
        return result
        
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

def bkgdist(logeval, lon, lat):
    # np.log(1e6) factor is because the background rate is given in 1/MeV not 1/TeV for some reason
    return np.log(bkgfull.evaluate(energy=10**logeval*u.TeV, fov_lon=np.abs(lon)*u.deg, fov_lat=np.abs(lat)*u.deg).value*1e6/(2*np.pi))

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



from gammapy.astro.darkmatter import (
    profiles,
    JFactory
)

profile = profiles.EinastoProfile()

# Adopt standard values used in HESS
profiles.DMProfile.DISTANCE_GC = 8.5 * u.kpc
profiles.DMProfile.LOCAL_DENSITY = 0.39 * u.Unit("GeV / cm3")

profile.scale_to_local_density()

position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
geom = WcsGeom.create(skydir=position, 
                      binsz=longitudeaxistrue[1]-longitudeaxistrue[0],
                      width=(longitudeaxistrue[-1]-longitudeaxistrue[0]+longitudeaxistrue[1]-longitudeaxistrue[0], latitudeaxistrue[-1]-latitudeaxistrue[0]+longitudeaxistrue[1]-longitudeaxistrue[0]),
                      frame="galactic")


jfactory = JFactory(
    geom=geom, profile=profile, distance=profiles.DMProfile.DISTANCE_GC
)
jfact = jfactory.compute_differential_jfactor().value


def setup_full_fake_signal_dist(logmass, specfunc):
    # numpy vectorisation
    def full_fake_signal_dist(log10eval, lonval, latval):
        log10eval = np.array(log10eval)
        # nicespatialfunc = stats.multivariate_normal(mean=[0,0], cov=[[1.0,0.0],[0.0,1.0]]).logpdf
        # nicespatialfunc = log_squared_einasto_lb
        if log10eval.ndim>1:
            spectralvals = np.squeeze(specfunc(logmass, log10eval[:,0, 0]))
            spatialvals = np.log(jfact.T)
            logpdfvalues = spectralvals[:, np.newaxis,np.newaxis]+spatialvals[np.newaxis, :,:]
            return logpdfvalues
        else:
            logpdfvalues = specfunc(logmass, log10eval)+jfact[np.where(longitudeaxistrue==lonval, latitudeaxistrue==latval)]
            return logpdfvalues
    
    return full_fake_signal_dist

def setup_full_fake_signal_dist_copy_vers(logmass, specfunc):
    # numpy vectorisation
    def full_fake_signal_dist(log10eval, lonval, latval):
        spectralvals = np.squeeze(specfunc(logmass, log10eval))
        spatialvals = np.log(jfact.T)
        logpdfvalues = spectralvals+spatialvals
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

# offsettruemeshpsf, offsetreconmeshpsf, logetruemeshpsf = np.meshgrid(offsetaxistrue, offsetaxis, log10eaxistrue)
# psfnormalisations = special.logsumexp(psf(offsetreconmeshpsf.flatten(), offsettruemeshpsf.flatten(), logetruemeshpsf.flatten()).reshape(offsetreconmeshpsf.shape),axis=0)

# logetruemeshedisp, logereconmeshedisp, offsettruemeshedisp,  = np.meshgrid(log10eaxistrue, log10eaxis, offsetaxistrue)
# edispnormalisations = special.logsumexp(edisp(logereconmeshedisp.flatten(), logetruemeshedisp.flatten(), offsettruemeshedisp.flatten()).reshape(logereconmeshedisp.shape).T +logjacob,axis=2)


# def calcirfvals(measuredcoord, log10eaxis=log10eaxis, spatialaxistrue=spatialaxistrue):
#     logemeasured, offsetmeasured    = measuredcoord
#     log10emesh, offsetmesh          = np.meshgrid(log10eaxistrue, spatialaxistrue)
#     energyloglikelihoodvals         = edisp(logemeasured, log10emesh, offsetmesh)#-edispnormalisations
#     pointspreadlikelihoodvals       = psf(offsetmeasured, offsetmesh, log10emesh)#-psfnormalisations
    
    
#     return energyloglikelihoodvals+pointspreadlikelihoodvals


# def calcdemderirfvals(datatuple, lontrue_mesh_nuisance, logetrue_mesh_nuisance, lattrue_mesh_nuisance, edispnormalisation=0,  psfnormalisation=0):
#     logeval, coord = datatuple
    
#     return psf(coord, np.array([lontrue_mesh_nuisance.flatten(), lattrue_mesh_nuisance.flatten()]), logetrue_mesh_nuisance.flatten()).reshape(logetrue_mesh_nuisance.shape)+\
#         edisp(logeval, logetrue_mesh_nuisance.flatten(), np.array([lontrue_mesh_nuisance.flatten(), lattrue_mesh_nuisance.flatten()])).reshape(logetrue_mesh_nuisance.shape) - edispnormalisation - psfnormalisation

def calcrirfindices(datatuple):
    logeval, coord = datatuple
    
    log10eindex = np.squeeze(np.where(log10eaxis==logeval))
    longitude_index = np.squeeze(np.where(longitudeaxis==coord[0]))
    latitude_index = np.squeeze(np.where(latitudeaxis==coord[1]))

    
    return [log10eindex, longitude_index, latitude_index]


# Slower version of the marginalisation function as it only takes in
def marginalisenuisance(irfindices, prior, edispmatrix, psfmatrix):
    prior = prior - special.logsumexp(logjacobtrue+prior.T)
    marginalisationvalues = special.logsumexp(logjacobtrue+prior.T+edispmatrix[:,:,:,irfindices[0]].T+psfmatrix[:,:,:,irfindices[1],irfindices[2]].T)
    
    return marginalisationvalues


def evaluateformass(logmass, lambdarange, bkgmargvals, irfindexlist, specfunc, edispmatrix, psfmatrix, lontrue_mesh_nuisance, logetrue_mesh_nuisance, lattrue_mesh_nuisance):
    priorvals= setup_full_fake_signal_dist(logmass, specfunc=specfunc)(logetrue_mesh_nuisance, lontrue_mesh_nuisance, lattrue_mesh_nuisance)
    
    priornormalisation = special.logsumexp(priorvals.T+logjacobtrue)
    
    priorvals = priorvals.T-priornormalisation
        
    signalmarginalisationvalues = special.logsumexp(logjacobtrue+priorvals+edispmatrix[:,:,:,irfindexlist[:,0]].T+psfmatrix[:,:,:,irfindexlist[:,1],irfindexlist[:,2]].T, axis=(1,2,3))
    
    del priorvals
    
    gc.collect()
    # print(signalmarginalisationvalues.shape)
    
    lambdaoutput = np.sum(np.logaddexp(np.log(lambdarange[:, np.newaxis])+signalmarginalisationvalues[np.newaxis,:], np.log(1-lambdarange[:, np.newaxis])+bkgmargvals[np.newaxis,:]),axis=1)
    
    del signalmarginalisationvalues
    gc.collect()
    
        
    return lambdaoutput



import matplotlib.transforms as transforms
import matplotlib.patches as patches

def confidence_ellipse(x, y, probabilities, ax, n_std=3.0, edgecolor='white',facecolor='none', **kwargs):
    
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

import functools

def sigmarg(logmass, specfunc, irfindexlist, edispmatrix, psfmatrix, logetrue_mesh_nuisance, lontrue_mesh_nuisance, lattrue_mesh_nuisance, logjacobtrue=logjacobtrue):
    priorvals= setup_full_fake_signal_dist(logmass, specfunc=specfunc)(logetrue_mesh_nuisance, lontrue_mesh_nuisance, lattrue_mesh_nuisance)
        

        
    tempsigmargfunc = functools.partial(marginalisenuisance, prior=priorvals, edispmatrix=edispmatrix, psfmatrix=psfmatrix)
    result = [tempsigmargfunc(singleeventindices) for singleeventindices in irfindexlist]
    
    return result


def diff_irf_marg(single_event_measurement, signal_prior_matrices, logbkgpriorvalues, logmassrange, edispnormvalues, psfnormvalues):
    
    single_measured_log10e, single_measured_lon, single_measured_lat = single_event_measurement

    logbkgpriorvalues = logbkgpriorvalues - special.logsumexp(logbkgpriorvalues.T+logjacobtrue)

    
    sigmargresults = []
    
    nuisance_logemesh, nuisance_longitudemesh, nuisance_latitudemesh, measured_logemesh, measured_longitudemesh, measured_latitudemesh  = np.meshgrid(log10eaxistrue, 
                                                                                    longitudeaxistrue, latitudeaxistrue, 
                                                                                        single_measured_log10e, single_measured_lon, single_measured_lat, indexing='ij')

    truecoords = np.array([nuisance_longitudemesh.flatten(), nuisance_latitudemesh.flatten()])
    reconcoords = np.array([measured_longitudemesh.flatten(), measured_latitudemesh.flatten()])
    

    rad = angularseparation(reconcoords, truecoords).flatten()
    offset  = convertlonlat_to_offset(truecoords).flatten()

    
    psfvalues = np.squeeze(psf_efficient(rad,
            nuisance_logemesh.flatten(),
            offset).reshape(nuisance_logemesh.shape)) - psfnormvalues

    edispvalues = np.squeeze(edisp_efficient(measured_logemesh.flatten(), 
                nuisance_logemesh.flatten(),
                offset).reshape(nuisance_logemesh.shape)) - edispnormvalues
    
    bkgmargresult = special.logsumexp(logbkgpriorvalues.T+logjacobtrue+psfvalues.T+edispvalues.T)

    
    singlemass_sigmargvals = []
    for logmass_idx, logmass in enumerate(logmassrange):
        
        output = special.logsumexp(signal_prior_matrices[logmass_idx]+logjacobtrue+psfvalues.T+edispvalues.T)#-irfnormalisation)
        singlemass_sigmargvals.append(output)

    sigmargresults.append(singlemass_sigmargvals)
        

        
    return np.array([np.squeeze(np.array(sigmargresults)), bkgmargresult])






