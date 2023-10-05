import numpy as np
from os import path
from .utils.utils import log10eaxistrue, convertlonlat_to_offset, angularseparation, longitudeaxistrue, latitudeaxistrue, psf_efficient, edisp_test, edisp_migra, log10eaxis, edisp_efficient
from scipy import interpolate
resources_dir = path.join(path.dirname(__file__), 'package_data')


astrophysicalbackground = np.load(resources_dir+"/unnormalised_astrophysicalbackground.npy")
psfnormalisationvalues = np.load(resources_dir+"/psfnormalisation.npy")
edispnormalisationvalues = np.load(resources_dir+"/edispnormalisation.npy")


true_offset_axis = np.linspace(0,convertlonlat_to_offset(np.array([longitudeaxistrue[-1], latitudeaxistrue[-1]])),71)

rad_axis = np.logspace(-3,0.5,71)
rad_axis = np.insert(rad_axis, 0, 0)

psf_mesh = np.meshgrid(rad_axis, log10eaxistrue, true_offset_axis, indexing='ij')

psfvals = psf_efficient(*psf_mesh)


psf_efficient_interpolator_tuple_input = interpolate.RegularGridInterpolator((rad_axis, log10eaxistrue, true_offset_axis,), 
                                                                             np.exp(psfvals), method='nearest', bounds_error=False, 
                                                                             fill_value=0)



# migraaxis = np.logspace(-1,0.5, 91)

# edisp_mesh = np.meshgrid(migraaxis, log10eaxistrue, true_offset_axis, indexing='ij')


# edispvals = edisp_migra(*edisp_mesh)


edisp_mesh = np.meshgrid(log10eaxis, log10eaxistrue, true_offset_axis, indexing='ij')


edispvals = edisp_efficient(*edisp_mesh)


edisp_efficient_interpolator = interpolate.RegularGridInterpolator((log10eaxis, log10eaxistrue, true_offset_axis,), np.exp(edispvals), method='nearest', bounds_error=False, 
                                                                 fill_value=0)

def log_single_likelihood(reconloge, recon_lon, recon_lat, logetrue, true_lon, true_lat):
    true_offset = convertlonlat_to_offset(np.array([true_lon, true_lat]),)
    rad = angularseparation(np.array([recon_lon, recon_lat]),np.array([true_lon, true_lat]))
    # migra = 10**(reconloge-logetrue)

    return edisp_efficient_interpolator((reconloge, logetrue, true_offset,))+psf_efficient_interpolator_tuple_input((rad, logetrue, true_offset,))