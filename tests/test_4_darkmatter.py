import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

import os, sys, time
from tqdm import tqdm

from gammabayes.likelihoods.irfs import IRF_LogLikelihood

from gammabayes.dark_matter.spectral_models import Z2_ScalarSinglet
from gammabayes.dark_matter import CombineDMComps
from gammabayes.dark_matter.density_profiles import Einasto_Profile

import numpy as np
from astropy import units as u


def test_dm_spectral_cutoff():
    energy_true_axis, longitudeaxistrue, latitudeaxistrue = np.logspace(-1,2,31)*u.TeV, np.linspace(-5,5,20)*u.deg, np.linspace(-4,4,16)*u.deg

    energy_recon_axis, longitudeaxis, latitudeaxis = np.logspace(-1,2,16)*u.TeV, np.linspace(-5,5,10)*u.deg, np.linspace(-4,4,8)*u.deg


    irf_loglike = IRF_LogLikelihood(axes=[energy_recon_axis, longitudeaxis, latitudeaxis], 
                                    dependent_axes=[energy_true_axis, longitudeaxistrue, latitudeaxistrue])

    logDMprior = CombineDMComps(name='Z2 Scalar Singlet dark matter',
                            spectral_class = Z2_ScalarSinglet, 
                            spatial_class = Einasto_Profile,
                            irf_loglike=irf_loglike, 
                            axes=(energy_true_axis, 
                                longitudeaxistrue, 
                                latitudeaxistrue,), 
                            default_spectral_parameters={'mass':1.0,}, )
    assert np.isneginf(logDMprior(2.0*u.TeV, 0.1*u.deg, 0.0*u.deg,spectral_parameters={'mass':1.0}))

