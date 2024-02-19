import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

import os, sys, time
from tqdm import tqdm

from gammabayes.likelihoods.irfs import IRF_LogLikelihood

from gammabayes.dark_matter.spectral_models import Z2_ScalarSinglet
from gammabayes.dark_matter import CombineDMComps
from gammabayes.dark_matter.density_profiles import Einasto_Profile

import numpy as np


def test_dm_spectral_cutoff():
    energy_true_axis, longitudeaxistrue, latitudeaxistrue = np.logspace(-1,2,31), np.linspace(-5,5,20), np.linspace(-4,4,16)

    energy_recon_axis, longitudeaxis, latitudeaxis = np.logspace(-1,2,16), np.linspace(-5,5,10), np.linspace(-4,4,8)


    irf_loglike = IRF_LogLikelihood(axes=[energy_recon_axis, longitudeaxis, latitudeaxis], 
                                    dependent_axes=[energy_true_axis, longitudeaxistrue, latitudeaxistrue])

    logDMprior = CombineDMComps(name='Z2 Scalar Singlet dark matter',
                            spectral_class = Z2_ScalarSinglet, 
                            spatial_class = Einasto_Profile,
                            irf_loglike=irf_loglike, 
                            axes=(energy_true_axis, 
                                longitudeaxistrue, 
                                latitudeaxistrue,), 
                            axes_names=['energy', 'lon', 'lat'],
                            default_spectral_parameters={'mass':1.0,}, )
    assert np.isneginf(logDMprior(2.0, 0.00001, 0.0,spectral_parameters={'mass':1.0}))

