import numpy as np
from os import path
from gammabayes.utils.utils import resources_dir
log_psf_normalisations = np.load(resources_dir+"/log_prod5_psf_normalisations.npy")
log_edisp_normalisations = np.load(resources_dir+"/log_prod5_edisp_normalisations.npy")