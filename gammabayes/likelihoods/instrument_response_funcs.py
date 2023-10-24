from ..utils.utils import convertlonlat_to_offset, angularseparation, resource_dir
import numpy as np
from astropy import units as u
from gammapy.irf import load_cta_irfs
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom





np.seterr(divide = 'ignore')
# I believe this is the alpha configuration of the array as there are no LSTs
irfs = load_cta_irfs(resource_dir+'/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits')


edispfull = irfs['edisp']
psffull = irfs['psf']
edispfull.normalize()
bkgfull = irfs['bkg']
psf3d = psffull.to_psf3d()
aefffull = irfs['aeff']

aefffunc = lambda energy, offset: aefffull.evaluate(energy_true = energy*u.TeV, offset=offset*u.deg).to(u.cm**2).value

def log_aeff(logetrue, true_lon, true_lat):
    return np.log(aefffull.evaluate(energy_true = 10**logetrue*u.TeV, 
                             offset=convertlonlat_to_offset(np.array([true_lon, true_lat]))*u.deg).to(u.m**2).value)
    
def log_edisp(reconloge, logetrue, true_lon, true_lat):
    return np.log(edispfull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                    migra = np.power(10.,reconloge-logetrue), 
                                                    offset=convertlonlat_to_offset(np.array([true_lon, true_lat]))*u.deg).value)


def log_psf(recon_lon, recon_lat, logetrue, true_lon, true_lat):
    reconstructed_spatialcoord = np.array([recon_lon, recon_lat])
    truespatialcoord = np.array([true_lon, true_lat])
    rad = angularseparation(reconstructed_spatialcoord, truespatialcoord).flatten()
    offset  = convertlonlat_to_offset(truespatialcoord).flatten()
    output = np.log(psffull.evaluate(energy_true=10**logetrue*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    return output


def single_loglikelihood(reconloge, recon_lon, recon_lat, logetrue, true_lon, true_lat):
    reconstructed_spatialcoord = np.array([recon_lon, recon_lat])
    truespatialcoord = np.array([true_lon, true_lat])
    rad = angularseparation(reconstructed_spatialcoord, truespatialcoord).flatten()
    offset  = convertlonlat_to_offset(truespatialcoord).flatten()
    output = np.log(psffull.evaluate(energy_true=10**logetrue*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    output+=np.log(edispfull.evaluate(energy_true=10**logetrue*u.TeV,
                                                    migra = 10**(reconloge-logetrue), 
                                                    offset=offset*u.deg).value)
    
    return output




def log_bkg_CCR_dist(logeval, lon, lat):
    # np.log(1e6) factor is because the background rate is given in 1/MeV not 1/TeV for some reason
    return np.log(bkgfull.evaluate(energy=10**logeval*u.TeV, fov_lon=np.abs(lon)*u.deg, fov_lat=np.abs(lat)*u.deg).value*1e6)


# Older testing functions

# def aeff_efficient(logetrue, offset):
#     return np.log(aefffull.evaluate(energy_true=10**logetrue*u.TeV, offset=offset*u.deg).to(u.cm**2).value)



# def psf_efficient(rad, logetrue, offset):

#     output = np.log(psffull.evaluate(energy_true=10**logetrue*u.TeV,
#                                                     rad = rad*u.deg, 
#                                                     offset=offset*u.deg).value)
    
#     return output


# def edisp_efficient(logereconstructed, logetrue, offset):
#     return np.log(edispfull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
#                                                     migra = np.power(10.,logereconstructed-logetrue), 
#                                                     offset=offset*u.deg).value)