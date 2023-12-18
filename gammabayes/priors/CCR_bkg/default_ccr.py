import numpy as np
from astropy import units as u
from gammapy.irf import load_irf_dict_from_file
from gammabayes.utils import convertlonlat_to_offset, angularseparation, resources_dir


irfs = load_irf_dict_from_file(resources_dir+'/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits')

bkgfull = irfs['bkg']


def log_bkg_CCR_dist(energyval, lon, lat, spectral_parameters={}, spatial_parameters={}):
    """Wrapper for the Gammapy interpretation of the log of 
        the CTA's background charged cosmic-ray mis-identification rate.

    Args:
        energyval (float): True energy of a gamma-ray event detected by the CTA
        lon (float): True FOV longitude of a gamma-ray event 
            detected by the CTA
        lat (float): True FOV latitude of a gamma-ray event 
            detected by the CTA

    Returns:
        float: Natural log of the charged cosmic ray mis-idenfitication rate for the CTA
    """
    return np.log(bkgfull.evaluate(energy=energyval*u.TeV, fov_lon=lon*u.deg, fov_lat=lat*u.deg).to((u.TeV*u.sr*u.s)**(-1)).value)
