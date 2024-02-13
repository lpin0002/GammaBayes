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
    energy_true_axis, longitudeaxistrue, latitudeaxistrue = np.logspace(-1,2,31), np.linspace(-5,5,21), np.linspace(-4,4,17)

    energy_recon_axis, longitudeaxis, latitudeaxis = np.logspace(-1,2,16), np.linspace(-5,5,11), np.linspace(-4,4,9)


    irf_like = IRF_LogLikelihood(axes=[energy_recon_axis, longitudeaxis, latitudeaxis], dependent_axes=[energy_true_axis, longitudeaxistrue, latitudeaxistrue])

    logDMprior = CombineDMComps(Z2_ScalarSinglet, Einasto_Profile, irf_like)

    assert np.isneginf(logDMprior(2.0, 0.0, 0.0,spectral_parameters={'mass':1.0}))
