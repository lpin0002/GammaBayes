import numpy as np
from os import path
from gammabayes.utils.utils import resources_dir
log_psf_normalisations = np.load(resources_dir+"/log_prod3b_psf_normalisations.npy")
log_edisp_normalisations = np.load(resources_dir+"/log_prod3b_edisp_normalisations.npy")