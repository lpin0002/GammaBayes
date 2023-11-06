import numpy as np
from os import path
resources_dir = path.join(path.dirname(path.dirname(__file__)), 'package_data')
print('resources_dir: ', resources_dir)

astrophysicalbackground = np.load(resources_dir+"/unnormalised_astrophysicalbackground.npy")
diffuse_astrophysicalbackground = np.load(resources_dir+"/unnormalised_astrophysical_diffuse_background.npy")
point_astrophysicalbackground = np.load(resources_dir+"/unnormalised_astrophysical_point_background.npy")
psfnormalisationvalues = np.load(resources_dir+"/psfnormalisation.npy")
edispnormalisationvalues = np.load(resources_dir+"/edispnormalisation.npy")