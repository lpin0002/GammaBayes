from scipy import integrate, special, interpolate, stats
import numpy as np
from tqdm import tqdm
from gammapy.irf import load_cta_irfs
from astropy import units as u

from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
import sys, yaml, pickle, os, random, time, warnings

from os import path
resources_dir = path.join(path.dirname(__file__), '../package_data')



np.seterr(divide = 'ignore')
# I believe this is the alpha configuration of the array as there are no LSTs
irfs = load_cta_irfs(resources_dir+'/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits')
edispfull = irfs['edisp']
psffull = irfs['psf']
edispfull.normalize()
bkgfull = irfs['bkg']
psf3d = psffull.to_psf3d()
aefffull = irfs['aeff']


def create_axis(lower_bound, upper_bound, number_of_bins_per_unit=100, resolution=None):
    if resolution is None:
        return np.linspace(lower_bound, 
                           upper_bound, 
                           int(round(number_of_bins_per_unit*(upper_bound-lower_bound))+1))
    else:
        return np.linspace(lower_bound, 
                         upper_bound, 
                         int(round((upper_bound-lower_bound)/resolution)+1))
    
def create_axes(log10_energy_min, log10_energy_max, 
                     log10_energy_bins_per_decade, spatial_res, 
                     longitude_min, longitude_max,
                     latitude_min, latitude_max):
    
    print(log10_energy_min, log10_energy_max, 
                     log10_energy_bins_per_decade, spatial_res, 
                     longitude_min, longitude_max,
                     latitude_min, latitude_max)
    
    log10_eaxis = create_axis(log10_energy_min, log10_energy_max, number_of_bins_per_unit=log10_energy_bins_per_decade)
    longitude_axis = create_axis(longitude_min, longitude_max, resolution=spatial_res)
    latitude_axis = create_axis(latitude_min, latitude_max, resolution=spatial_res)

    return log10_eaxis, longitude_axis, latitude_axis



def convertlonlat_to_offset(fov_coord):
    """Takes a coordinate and translates that into an offset assuming small angles

    Args:
        fov_coord (np.ndarray): A coordinate in FOV frame

    Returns:
        np.ndarray or float: The corresponding offset values for the given fov coordinates
            assuming small angles
    """
    
    return np.linalg.norm(fov_coord, axis=0)


def angularseparation(coord1, coord2):
    """Calculates the angular separation between coord1 and coord2 in FOV frame
        assuming small angles.

    Args:
        coord1 (np.ndarray): First coordinate in FOV frame
        coord2 (np.ndarray): Second coordinate in FOV frame

    Returns:
        float: Angular sepration between the two coords assuming small angles
    """    
    try:
        return np.linalg.norm(coord2-coord1, axis=0)
    except:
        try:
            return np.linalg.norm(coord2-coord1.T, axis=1)
        except:
            return np.linalg.norm(coord2.T-coord1, axis=1)







def makelogjacob(log10eaxis):
    """_summary_

    Args:
        log10eaxis (np.ndarray, optional): Axis of discrete values of log10 energy values. 
        Defaults to log10eaxis.

    Returns:
        np.ndarray: Log jacobian for using log10 energy to get integral over energy
    """
    outputlogjacob = np.log(10**log10eaxis)#+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0])
    return outputlogjacob




    
    
def edisp_test(reconloge, logetrue, true_lon, true_lat):
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

def psf_test(recon_lon, recon_lat, logetrue, true_lon, true_lat):
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




def single_likelihood(reconloge, recon_lon, recon_lat, logetrue, true_lon, true_lat):
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

def log_aeff(logetrue, longitude, latitude):
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
    offset  = convertlonlat_to_offset(np.array([longitude, latitude])).flatten()

    return np.log(aefffull.evaluate(energy_true=10**logetrue*u.TeV, offset=offset*u.deg).to(u.m**2).value)


def bkgdist(logeval, lon, lat):
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
    return np.log(bkgfull.evaluate(energy=10**logeval*u.TeV, fov_lon=np.abs(lon)*u.deg, fov_lat=np.abs(lat)*u.deg).to(1/(u.sr*u.s*u.TeV)).value)



