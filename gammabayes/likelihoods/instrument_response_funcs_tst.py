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
    """Wrapper for the Gammapy interpretation of the log of 
        the CTA effective area function.

    Args:
        logetrue (float): True energy of a gamma-ray event detected by the CTA
        true_lon (float): True FOV longitude of a gamma-ray event 
            detected by the CTA
        true_lat (float): True FOV latitude of a gamma-ray event 
            detected by the CTA

    Returns:
        float: The natural log of the effective area of the CTA in m^2
    """
    return np.log(aefffull.evaluate(energy_true = 10**logetrue*u.TeV, 
                             offset=convertlonlat_to_offset(np.array([true_lon, true_lat]))*u.deg).to(u.m**2).value)
    
def log_edisp(reconloge, logetrue, true_lon, true_lat):
    """Wrapper for the Gammapy interpretation of the CTA point spread function.

    Args:
        reconloge (float): Measured energy value by the CTA
        logetrue (float): True energy of a gamma-ray event detected by the CTA
        true_lon (float): True FOV longitude of a gamma-ray event 
            detected by the CTA
        true_lat (_type_): True FOV latitude of a gamma-ray event 
            detected by the CTA

    Returns:
        float: natural log of the CTA energy dispersion likelihood for the given 
            gamma-ray event data
    """
    return np.log(edispfull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                    migra = np.power(10.,reconloge-logetrue), 
                                                    offset=convertlonlat_to_offset(np.array([true_lon, true_lat]))*u.deg).value)


def log_psf(recon_lon, recon_lat, logetrue, true_lon, true_lat):
    """Wrapper for the Gammapy interpretation of the CTA point spread function.

    Args:
        recon_lon (float): Measured FOV longitude of a gamma-ray event
            detected by the CTA
        recon_lat (float): Measured FOV latitude of a gamma-ray event
            detected by the CTA
        logetrue (float): True energy of a gamma-ray event detected by the CTA
        true_lon (float): True FOV longitude of a gamma-ray event 
            detected by the CTA
        true_lat (float): True FOV latitude of a gamma-ray event 
            detected by the CTA

    Returns:
        float: natural log of the CTA point spread function likelihood for the given 
            gamma-ray event data
    """
    reconstructed_spatialcoord = np.array([recon_lon, recon_lat])
    truespatialcoord = np.array([true_lon, true_lat])
    rad = angularseparation(reconstructed_spatialcoord, truespatialcoord).flatten()
    offset  = convertlonlat_to_offset(truespatialcoord).flatten()
    output = np.log(psffull.evaluate(energy_true=10**logetrue*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    return output


def single_loglikelihood(reconloge, recon_lon, recon_lat, logetrue, true_lon, true_lat):
    """Wrapper for the Gammapy interpretation of the CTA IRFs to output the log 
        likelihood values for the given gamma-ray event data

    Args:
        reconloge (float): Measured energy value by the CTA
        recon_lon (float): Measured FOV longitude of a gamma-ray event
            detected by the CTA
        recon_lat (float): Measured FOV latitude of a gamma-ray event
            detected by the CTA
        logetrue (float): True energy of a gamma-ray event detected by the CTA
        true_lon (float): True FOV longitude of a gamma-ray event 
            detected by the CTA
        true_lat (float): True FOV latitude of a gamma-ray event 
            detected by the CTA

    Returns:
        float: natural log of the full CTA likelihood for the given gamma-ray 
            event data
    """
    psf_value_mesh = np.meshgrid(np.unique(recon_lon).flatten(), 
        np.unique(recon_lat), np.unique(logetrue).flatten(), np.unique(true_lon).flatten(), np.unique(true_lat).flatten(), indexing='ij')

    psf_reconstructed_spatialcoord = np.array([psf_value_mesh[0].flatten(), psf_value_mesh[1].flatten()])
    psf_truespatialcoord = np.array([psf_value_mesh[-2].flatten(), psf_value_mesh[-1].flatten()])
    psf_rad = angularseparation(psf_reconstructed_spatialcoord, psf_truespatialcoord).flatten()

    psf_offset  = convertlonlat_to_offset(psf_truespatialcoord).flatten()
    psf_output = np.log(psffull.evaluate(energy_true=10**psf_value_mesh[2].flatten()*u.TeV,
                                                    rad = psf_rad*u.deg, 
                                                    offset=psf_offset*u.deg).value).reshape(psf_value_mesh[0].shape)

    edisp_value_mesh = np.meshgrid(np.unique(reconloge), np.unique(logetrue), 
        np.unique(true_lon), np.unique(true_lat), indexing='ij')

    edisp_truespatialcoord = np.array([edisp_value_mesh[-2].flatten(), edisp_value_mesh[-1].flatten()])

    edisp_offset  = convertlonlat_to_offset(edisp_truespatialcoord).flatten()
    
    edisp_output =np.log(edispfull.evaluate(energy_true=10**edisp_value_mesh[1].flatten()*u.TeV,
                                                    migra = 10**(edisp_value_mesh[0].flatten()-edisp_value_mesh[1].flatten()), 
                                                    offset=edisp_offset*u.deg).value).reshape(edisp_value_mesh[0].shape)

    output = edisp_output[:,np.newaxis,np.newaxis, :,:,:]+psf_output[np.newaxis,:,:,:,:,:]
    
    return output




def log_bkg_CCR_dist(logeval, lon, lat):
    """Wrapper for the Gammapy interpretation of the log of 
        the CTA's background charged cosmic-ray mis-identification rate.

    Args:
        logeval (float): True energy of a gamma-ray event detected by the CTA
        lon (float): True FOV longitude of a gamma-ray event 
            detected by the CTA
        lat (float): True FOV latitude of a gamma-ray event 
            detected by the CTA

    Returns:
        float: Natural log of the charged cosmic ray mis-idenfitication rate for the CTA
    """
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