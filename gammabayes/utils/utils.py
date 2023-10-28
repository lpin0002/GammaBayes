from scipy import integrate, special, interpolate, stats
import numpy as np
import random, time
from tqdm import tqdm

import yaml, warnings, sys, os


from os import path
resource_dir = path.join(path.dirname(__file__), '../package_data')

def convertlonlat_to_offset(fov_coord):
    """Takes a coordinate and translates that into an offset assuming small angles

    Args:
        fov_coord (np.ndarray): A coordinate in FOV frame

    Returns:
        np.ndarray or float: The corresponding offset values for the given fov coordinates
            assuming small angles
    """
    return np.linalg.norm(fov_coord, axis=0)


def angularseparation(coord1, coord2=None):
    """Calculates the angular separation between coord1 and coord2 in FOV frame
        assuming small angles.

    Args:
        coord1 (np.ndarray): First coordinate in FOV frame
        coord2 (np.ndarray): Second coordinate in FOV frame

    Returns:
        float: Angular sepration between the two coords assuming small angles
    """    
    
    try:
        return np.linalg.norm(coord2-coord1, axis=0)
    except:
        try:
            return np.linalg.norm(coord2-coord1.T, axis=1)
        except:
            return np.linalg.norm(coord2.T-coord1, axis=1)


def bin_centres_to_edges(axis):
    return np.append(axis-np.diff(axis)[0]/2, axis[-1]+np.diff(axis)[0]/2)