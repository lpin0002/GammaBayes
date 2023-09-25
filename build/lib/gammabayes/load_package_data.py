import numpy as np
from os import path
resources_dir = path.join(path.dirname(__file__), 'package_data')


astrophysicalbackground = np.load(resources_dir+"/unnormalised_astrophysicalbackground.npy")
psfnormalisationvalues = np.load(resources_dir+"/psfnormalisation.npy")
edispnormalisationvalues = np.load(resources_dir+"/edispnormalisation.npy")