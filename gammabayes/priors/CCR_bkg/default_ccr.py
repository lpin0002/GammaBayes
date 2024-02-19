import numpy as np
from astropy import units as u
from gammapy.irf import load_irf_dict_from_file
from gammabayes.utils import resources_dir


irfs = load_irf_dict_from_file(resources_dir+'/irf_fits_files/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits')

bkgfull = irfs['bkg']


def log_bkg_CCR_dist(energy, longitude, latitude, spectral_parameters={}, spatial_parameters={}):
    """Wrapper for the Gammapy interpretation of the log of 
        the CTA's background charged cosmic-ray mis-identification rate.

    Args:
        energy (float): True energy of a gamma-ray event detected by the CTA
        longitude (float): True FOV longitude of a gamma-ray event 
            detected by the CTA
        latitude (float): True FOV latitude of a gamma-ray event 
            detected by the CTA

    Returns:
        float: Natural log of the charged cosmic ray mis-idenfitication rate for the CTA
    """
    return np.log(bkgfull.evaluate(energy=energy*u.TeV, fov_lon=longitude*u.deg, fov_lat=latitude*u.deg).to((u.TeV*u.sr*u.s)**(-1)).value)
