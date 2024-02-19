import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

import os, sys, time
from tqdm import tqdm

from gammabayes.likelihoods.irfs import IRF_LogLikelihood

import numpy as np

def test_irf_input_output():
    energy_true_axis, longitudeaxistrue, latitudeaxistrue = np.logspace(-1,2,31), np.linspace(-5,5,21), np.linspace(-4,4,17)

    energy_recon_axis, longitudeaxis, latitudeaxis = np.logspace(-1,2,16), np.linspace(-5,5,11), np.linspace(-4,4,9)


    irf_loglike = IRF_LogLikelihood(axes=[energy_recon_axis, longitudeaxis, latitudeaxis], 
                                    dependent_axes = [energy_true_axis, longitudeaxistrue, latitudeaxistrue],
                                    pointing_direction=np.asarray([10,10]).T, 
                                    zenith=40, 
                                    hemisphere='North', 
                                    prod_vers=5)
    recon_lon, recon_lat, true_energy, true_lon, true_lat = np.asarray(0.), np.asarray(0.), np.asarray(1.0), np.asarray(0.), np.asarray(0.)

    result = np.squeeze(irf_loglike.log_psf(recon_lon=recon_lon, recon_lat=recon_lat, true_energy=true_energy, true_lon=true_lon, true_lat=true_lat))
    print(result)
    assert np.isneginf(result)


    recon_lon, recon_lat, true_energy, true_lon, true_lat = np.asarray(10.), np.asarray(10.), np.asarray(1.0), np.asarray(10.), np.asarray(10.)

    result = np.squeeze(irf_loglike.log_psf(recon_lon=recon_lon, recon_lat=recon_lat, true_energy=true_energy, true_lon=true_lon, true_lat=true_lat))
    print(result)
    assert not(np.isneginf(result))

