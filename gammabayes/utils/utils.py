from scipy import integrate, special, interpolate, stats
import numpy as np
import random, time
from tqdm import tqdm

import yaml, warnings, sys, os


from os import path
resource_dir = path.join(path.dirname(__file__), '../package_data')

def convertlonlat_to_offset(fov_coord):
    # Currently assuming small angles (|angle|<=4)
    return np.linalg.norm(fov_coord, axis=0)


def angularseparation(coord1, coord2=None):
    # Currently assuming small angles (|angle|<=4)
    
    try:
        return np.linalg.norm(coord2-coord1, axis=0)
    except:
        try:
            return np.linalg.norm(coord2-coord1.T, axis=1)
        except:
            return np.linalg.norm(coord2.T-coord1, axis=1)


