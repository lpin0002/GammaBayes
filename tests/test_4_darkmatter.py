import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

import os, sys, time
from tqdm import tqdm

from gammabayes.likelihoods.irfs import irf_loglikelihood

from gammabayes.dark_matter.models import SS_Spectra
from gammabayes.dark_matter import combine_DM_models
from gammabayes.dark_matter.density_profiles import Einasto_Profile

import numpy as np


# def test_dm_spectral_cutoff():
#     energy_true_axis, longitudeaxistrue, latitudeaxistrue = np.logspace(-1,2,31), np.linspace(-5,5,21), np.linspace(-4,4,17)

#     energy_recon_axis, longitudeaxis, latitudeaxis = np.logspace(-1,2,16), np.linspace(-5,5,11), np.linspace(-4,4,9)


#     irf_like = irf_loglikelihood(axes=[energy_recon_axis, longitudeaxis, latitudeaxis], dependent_axes=[energy_true_axis, longitudeaxistrue, latitudeaxistrue])

#     SS_DM_combine_instance = combine_DM_models(SS_Spectra, Einasto_Profile, irf_like)
#     logDMpriorfunc, logDMpriorfunc_mesh_efficient = SS_DM_combine_instance.DM_signal_dist, SS_DM_combine_instance.DM_signal_dist_mesh_efficient

#     assert np.isneginf(logDMpriorfunc(2.0, 0.0, 0.0,spectral_parameters={'mass':1.0}))
