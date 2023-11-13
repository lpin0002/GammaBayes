from .gammapy_wrappers import log_aeff, log_edisp, log_psf, single_loglikelihood
import warnings
try:
    from .load_default_irf_norms import log_psf_normalisations, log_edisp_normalisations
except Exception as e:
    # Print the error message without raising the exception
    print(f"An error occurred: {e}")