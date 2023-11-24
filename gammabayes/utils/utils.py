from scipy import integrate, special, interpolate, stats
import numpy as np
import random, time
from tqdm import tqdm
from scipy.stats import norm as norm1d
import yaml, warnings, sys, os


from os import path
resources_dir = path.join(path.dirname(__file__), '../package_data')

def convertlonlat_to_offset(fov_coord, pointing_direction=None):
    """Takes a coordinate and translates that into an offset assuming small angles

    Args:
        fov_coord (np.ndarray): A coordinate in FOV frame
        point_direction (np.ndarray): Pointing direction of telescope

    Returns:
        np.ndarray or float: The corresponding offset values for the given fov coordinates
            assuming small angles
    """
    if pointing_direction is None:
        return np.linalg.norm(fov_coord, axis=0)
    else:
        return np.linalg.norm(fov_coord.T-pointing_direction.T, axis=1)


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


def hdp_credible_interval_1d(y, sigma, x):
    y = y/integrate.simps(y=y, x=x)
    levels = np.linspace(0, y.max(),1000)

    areas = integrate.simps(y= (y>=levels[:, None])*y, x=x, axis=1)

    interpolator = interpolate.interp1d(y=levels, x=areas)
    if sigma!=0:
        prob_val = norm1d.cdf(sigma)-norm1d.cdf(-sigma)

        level = interpolator(prob_val)

        prob_array_indices = np.where(y>=level)

        return x[prob_array_indices[0][0]],x[prob_array_indices[0][-1]]
    else:
        cdf = integrate.cumtrapz(y=y, x=x)

        probval = norm1d.cdf(sigma)

        probidx = np.argmax(cdf >= probval)

        return [x[probidx]]


def power_law(energy, index, phi0=1):
    return phi0*energy**(index)











