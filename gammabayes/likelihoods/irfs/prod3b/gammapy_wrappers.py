from gammabayes.utils import resources_dir, haversine
import numpy as np
from astropy import units as u
from gammapy.irf import load_cta_irfs
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom





np.seterr(divide = 'ignore')
# I believe this is the alpha configuration of the array as there are no LSTs
prod3birfs = load_cta_irfs(resources_dir+'/prod3b_v2_South_z20_50h_irf_file.fits')



edispfull = prod3birfs['edisp']
psffull = prod3birfs['psf']

psf3d = psffull.to_psf3d()
aefffull = prod3birfs['aeff']

aefffunc = lambda energy, offset: aefffull.evaluate(energy_true = energy*u.TeV, offset=offset*u.deg).to(u.cm**2).value

def log_aeff(true_energy, true_lon, true_lat, pointing_direction=[0,0]):
    """Wrapper for the Gammapy interpretation of the log of 
        the CTA effective area function.

    Args:
        true_energy (float): True energy of a gamma-ray event detected by the CTA
        true_lon (float): True FOV longitude of a gamma-ray event 
            detected by the CTA
        true_lat (float): True FOV latitude of a gamma-ray event 
            detected by the CTA

    Returns:
        float: The natural log of the effective area of the CTA in m^2
    """
    return np.log(aefffull.evaluate(energy_true = true_energy*u.TeV, 
                             offset=haversine(
                                 true_lon, true_lat, pointing_direction[0], pointing_direction[1])*u.deg).to(u.cm**2).value)
    
def log_edisp(recon_energy, true_energy, true_lon, true_lat, pointing_direction=[0,0]):
    """Wrapper for the Gammapy interpretation of the CTA point spread function.

    Args:
        recon_energy (float): Measured energy value by the CTA
        true_energy (float): True energy of a gamma-ray event detected by the CTA
        true_lon (float): True FOV longitude of a gamma-ray event 
            detected by the CTA
        true_lat (_type_): True FOV latitude of a gamma-ray event 
            detected by the CTA

    Returns:
        float: natural log of the CTA energy dispersion likelihood for the given 
            gamma-ray event data
    """
    return np.log(edispfull.evaluate(energy_true=true_energy*u.TeV,
                                                    migra = recon_energy/true_energy, 
                                                    offset=haversine(
                                 true_lon, true_lat, pointing_direction[0], pointing_direction[1])*u.deg).value)


def log_psf(recon_lon, recon_lat, true_energy, true_lon, true_lat, pointing_direction=[0,0]):
    """Wrapper for the Gammapy interpretation of the CTA point spread function.

    Args:
        recon_lon (float): Measured FOV longitude of a gamma-ray event
            detected by the CTA
        recon_lat (float): Measured FOV latitude of a gamma-ray event
            detected by the CTA
        true_energy (float): True energy of a gamma-ray event detected by the CTA
        true_lon (float): True FOV longitude of a gamma-ray event 
            detected by the CTA
        true_lat (float): True FOV latitude of a gamma-ray event 
            detected by the CTA

    Returns:
        float: natural log of the CTA point spread function likelihood for the given 
            gamma-ray event data
    """
    rad = haversine(recon_lon, recon_lat, true_lon, true_lat).flatten()
    offset  = haversine(true_lon, true_lat, pointing_direction[0], pointing_direction[1]).flatten()
    output = np.log(psffull.evaluate(energy_true=true_energy*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    return output


def single_loglikelihood(recon_energy, recon_lon, recon_lat, true_energy, true_lon, true_lat, pointing_direction=[0,0]):
    """Wrapper for the Gammapy interpretation of the CTA IRFs to output the log 
        likelihood values for the given gamma-ray event data

    Args:
        recon_energy (float): Measured energy value by the CTA
        recon_lon (float): Measured FOV longitude of a gamma-ray event
            detected by the CTA
        recon_lat (float): Measured FOV latitude of a gamma-ray event
            detected by the CTA
        true_energy (float): True energy of a gamma-ray event detected by the CTA
        true_lon (float): True FOV longitude of a gamma-ray event 
            detected by the CTA
        true_lat (float): True FOV latitude of a gamma-ray event 
            detected by the CTA

    Returns:
        float: natural log of the full CTA likelihood for the given gamma-ray 
            event data
    """
    reconstructed_spatialcoord = np.array([recon_lon, recon_lat])
    truespatialcoord = np.array([true_lon, true_lat])
    rad = haversine(reconstructed_spatialcoord[0], reconstructed_spatialcoord[1], 
                    truespatialcoord[0], truespatialcoord[1]).flatten()
    offset  = haversine(true_lon.flatten(), true_lat.flatten(), pointing_direction[0].flatten(), pointing_direction[1].flatten()).flatten()
    output  = np.log(psffull.evaluate(energy_true=true_energy*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    output  += np.log(edispfull.evaluate(energy_true=true_energy*u.TeV,
                                                    migra = recon_energy/true_energy, 
                                                    offset=offset*u.deg).value)
    return output



def dynesty_single_loglikelihood(true_vals, recon_energy, recon_lon, recon_lat, pointing_direction=None):
    """Wrapper for the Gammapy interpretation of the CTA IRFs to output the log 
        likelihood values for the given gamma-ray event data

    Args:
        true_energy (float): True energy of a gamma-ray event detected by the CTA
        true_lon (float): True FOV longitude of a gamma-ray event 
            detected by the CTA
        true_lat (float): True FOV latitude of a gamma-ray event 
            detected by the CTA
        recon_energy (float): Measured energy value by the CTA
        recon_lon (float): Measured FOV longitude of a gamma-ray event
            detected by the CTA
        recon_lat (float): Measured FOV latitude of a gamma-ray event
            detected by the CTA

    Returns:
        float: natural log of the full CTA likelihood for the given gamma-ray 
            event data
    """
    true_energy, true_lon, true_lat = true_vals
    reconstructed_spatialcoord = np.array([recon_lon, recon_lat])
    truespatialcoord = np.array([true_lon, true_lat])
    rad = haversine(reconstructed_spatialcoord[0], reconstructed_spatialcoord[1], truespatialcoord[0], truespatialcoord[1]).flatten()
    offset  = haversine(truespatialcoord[0].flatten(), truespatialcoord[1].flatten(), 
                        pointing_direction[0].flatten(), pointing_direction[1].flatten()).flatten()
    output  = np.log(psffull.evaluate(energy_true=true_energy*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    output  += np.log(edispfull.evaluate(energy_true=true_energy*u.TeV,
                                                    migra = recon_energy/true_energy, 
                                                    offset=offset*u.deg).value)
    return np.squeeze(output)