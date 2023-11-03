import numpy as np
from os import path
resources_dir = path.join(path.dirname(path.dirname(__file__)), 'package_data')
print('resources_dir: ', resources_dir)

astrophysicalbackground = np.load(resources_dir+"/unnormalised_astrophysicalbackground.npy")
psfnormalisationvalues = np.load(resources_dir+"/psfnormalisation.npy")
edispnormalisationvalues = np.load(resources_dir+"/edispnormalisation.npy")