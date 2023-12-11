from scipy import integrate, special, interpolate, stats
import numpy as np
import random, time
from tqdm import tqdm
from scipy.stats import norm as norm1d
import yaml, warnings, sys, os


from os import path
resources_dir = path.join(path.dirname(__file__), '../package_data')


def fill_missing_keys(dictionary, default_values):
    for key, default_value in default_values.items():
        dictionary[key] = dictionary.get(key, default_value)


def haversine(lon1, lat1, lon2, lat2):
    # Convert degrees to radians
    lon1, lat1 = lon1*np.pi/180, lat1*np.pi/180
    lon2, lat2 = lon2*np.pi/180, lat2*np.pi/180

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    angular_separation_rad = 2 * np.arcsin(np.sqrt(a))


    return angular_separation_rad*180/np.pi



def convertlonlat_to_offset(angular_coord: np.ndarray, pointing_direction: np.ndarray=np.array([0,0])) -> float|np.ndarray:
    """Takes a coordinate and translates that into an offset

    Args:
        angular_coord (np.ndarray): Angular coordinates
        point_direction (np.ndarray): Pointing direction of telescope

    Returns:
        np.ndarray or float: The corresponding offset values for the given fov coordinates
            assuming small angles
    """
    delta_y = angular_coord[1, :] - pointing_direction[1]
    delta_x = angular_coord[0, :] - pointing_direction[0]

    # Calculate the angular separation using arctangent
    angles = np.arctan2(delta_y, delta_x)

    return angles * 180 / np.pi



def angularseparation(coord1: np.ndarray, coord2: np.ndarray|None =None) -> float|np.ndarray:
    """Takes a coordinate and translates that into an offset

    Args:
        angular_coord (np.ndarray): Angular coordinates
        point_direction (np.ndarray): Pointing direction of telescope

    Returns:
        np.ndarray or float: The corresponding offset values for the given fov coordinates
            assuming small angles
    """
    
    delta_y = coord1[1, :] - coord2[1, :]
    delta_x = coord1[0, :] - coord2[0, :]

    # Calculate the angular separation using arctangent
    angles = np.arctan2(delta_y, delta_x)

    return angles * 180 / np.pi


def bin_centres_to_edges(axis: np.ndarray) -> np.ndarray:
    return np.append(axis-np.diff(axis)[0]/2, axis[-1]+np.diff(axis)[0]/2)


def hdp_credible_interval_1d(y: np.ndarray, sigma: np.ndarray|list, x: np.ndarray) -> list[float, float]|list[float]:
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


def power_law(energy: np.ndarray|float, index: float, phi0: int =1) -> np.ndarray|float:
    return phi0*energy**(index)











