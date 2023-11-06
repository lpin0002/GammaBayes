import numpy as np
from os import path
from gammabayes.utils.utils import resources_dir
psfnormalisationvalues = np.load(resources_dir+"/psfnormalisation.npy")
edispnormalisationvalues = np.load(resources_dir+"/edispnormalisation.npy")