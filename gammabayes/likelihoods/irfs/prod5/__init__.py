from .gammapy_wrappers import *
from .irf_normalisation_setup import irf_norm_setup
from .irf_loglike_class import irf_loglikelihood
import warnings
try:
    from .load_irf_norms import log_psf_normalisations, log_edisp_normalisations
except Exception as e:
    # Print the error message without raising the exception
    print(f"An error occurred: {e}")