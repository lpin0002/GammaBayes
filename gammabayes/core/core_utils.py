from scipy import integrate, special, interpolate, stats
import numpy as np
import random, time, pickle
from tqdm import tqdm
from scipy.stats import norm as norm1d
import yaml, warnings, sys, os
from scipy.interpolate import RegularGridInterpolator
from astropy import units as u

from os import path
resources_dir = path.join(path.dirname(__file__), '../package_data')


def update_with_defaults(target_dict, default_dict):
    """
    Updates the target dictionary in place, adding missing keys from the default dictionary.

    Args:
        target_dict (dict): The dictionary to be updated.
        default_dict (dict): The dictionary containing default values.
    """
    for key, value in default_dict.items():
        target_dict.setdefault(key, value)


def haversine(lon1, lat1, lon2, lat2):
    # Convert degrees to radians
    lon1, lat1 = lon1.to(u.rad), lat1.to(u.rad)
    lon2, lat2 = lon2.to(u.rad), lat2.to(u.rad)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    angular_separation_rad = 2 * np.arcsin(np.sqrt(a))


    return angular_separation_rad.to(u.deg)



# def convertlonlat_to_offset(angular_coord: np.ndarray, pointing_direction: np.ndarray=np.array([0,0])) -> float|np.ndarray:
#     """Takes a coordinate and translates that into an offset

#     Args:
#         angular_coord (np.ndarray): Angular coordinates
#         point_direction (np.ndarray): Pointing direction of telescope

#     Returns:
#         np.ndarray or float: The corresponding offset values for the given fov coordinates
#             assuming small angles
#     """
#     delta_y = angular_coord[1, :] - pointing_direction[1]
#     delta_x = angular_coord[0, :] - pointing_direction[0]

#     # Calculate the angular separation using arctangent
#     angles = np.arctan2(delta_y, delta_x)

#     return angles * 180 / np.pi



# def angularseparation(coord1: np.ndarray, coord2: np.ndarray|None =None) -> float|np.ndarray:
#     """Takes a coordinate and translates that into an offset

#     Args:
#         angular_coord (np.ndarray): Angular coordinates
#         point_direction (np.ndarray): Pointing direction of telescope

#     Returns:
#         np.ndarray or float: The corresponding offset values for the given fov coordinates
#             assuming small angles
#     """
    
#     delta_y = coord1[1, :] - coord2[1, :]
#     delta_x = coord1[0, :] - coord2[0, :]

#     # Calculate the angular separation using arctangent
#     angles = np.arctan2(delta_y, delta_x)

#     return angles * 180 / np.pi





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





def save_to_pickle(filename, object_to_save):
    with open(filename, 'wb') as file:
        pickle.dump(object_to_save, file)


def load_pickle(filename):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)

    return loaded_object



def generate_unique_int_from_string(string):
    nums = []
    for character in string:
        num  = ord(character)
        nums.append(num)

    # List of the first 100 prime numbers, please don't judge me for hardcoding this
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
                       53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 
                       109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 
                       173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 
                       233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 
                       293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 
                       367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 
                       433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 
                       499, 503, 509, 521, 523, 541])
    primes_nums_length = primes[:len(nums)]

    integer = np.sum(nums*primes_nums_length)
    print(f"Unique int: {integer}")

    return integer


def extract_axes(axes_config):
    axes = {}

    for prior_axes in axes_config.values():
        for axes_for_input_type in prior_axes.values():
            for axis in axes_for_input_type.items():
                axes.update({axis[0]:axis[1]})

    return axes


def apply_dirichlet_stick_breaking_direct(mixtures_fractions: list | tuple, 
                                            depth: int) -> np.ndarray | float:

    dirichletmesh = 1

    for _dirichlet_i in range(depth):
        dirichletmesh*=(1-mixtures_fractions[_dirichlet_i])

    not_max_depth = depth!=len(mixtures_fractions)
    
    if not_max_depth:
        dirichletmesh*=mixtures_fractions[depth]

    return dirichletmesh


def _event_ratios_to_sticking_breaking_ratios(event_ratios: list | tuple) -> np.ndarray | float:
    event_ratios = np.asarray(event_ratios)


    stick_ratios = []

    for depth, event_ratio in enumerate(event_ratios):
        stick_ratio = event_ratio/(1-np.sum(event_ratios[:depth]))
        stick_ratios.append(stick_ratio)


    return np.asarray(stick_ratios)



def bound_axis(axis: np.ndarray, 
                bound_type: str, 
                bound_radii: float, 
                estimated_val: float):
    
    try:
        axis_unit = axis.unit
        axis = axis.value
    except:
        axis_unit = 1.


    try:
        bound_radii = bound_radii.to(axis_unit)
        estimated_val = estimated_val.to(axis_unit)

        bound_radii = bound_radii.value
        estimated_val = estimated_val.value
    except:
        pass



    if bound_type=='linear':
        axis_indices = np.where(
        (axis>estimated_val-bound_radii) & (axis<estimated_val+bound_radii) )[0]

    elif bound_type=='log10':
        axis_indices = np.where(
        (np.log10(axis)>np.log10(estimated_val)-bound_radii) & (np.log10(axis)<np.log10(estimated_val)+bound_radii) )[0]
        
    temp_axis = axis[axis_indices]*axis_unit

    return temp_axis, axis_indices




def bin_centres_to_edges(axis: np.ndarray) -> np.ndarray:
    return np.append(axis-np.diff(axis)[0]/2, axis[-1]+np.diff(axis)[0]/2)

# Currently unused
class FlexibleInterpolator:
    """
    A flexible wrapper for scipy's RegularGridInterpolator or similar interpolator functions, 
    allowing for interpolation over a grid of parameters with the capability to specify default values 
    for missing parameters during interpolation calls.

    Attributes:
        parameter_order (list): The order of parameters as they are expected by the interpolator.
        interpolation_parameter_points (list of arrays): The grid points for each parameter, corresponding to the order in parameter_order.
        values (ndarray): The values at each point in the grid to interpolate.
        default_values (dict): Default values for each parameter to use if a parameter is not specified during an interpolation call.
        interpolator_function (function): The interpolation function to use, defaulting to RegularGridInterpolator.
        interpolator_kwargs (dict): Additional keyword arguments to pass to the interpolator function.

    Methods:
        create_interpolator(**kwargs): Creates the interpolator instance using the provided grid points and values.
        __call__(**kwargs): Calls the interpolator with a set of parameters, using default values where necessary.
    """

    def __init__(self, parameter_order, interpolation_parameter_points, 
                 values, default_values, interpolator_function=RegularGridInterpolator,
                 **interpolator_kwargs):
        """
        Initializes the FlexibleInterpolator with the necessary parameters and creates the interpolator.

        Args:
            parameter_order (list): The order of parameters for interpolation.
            interpolation_parameter_points (list of arrays): The grid points for each parameter in the order specified by parameter_order.
            values (ndarray): The values at each point in the grid.
            default_values (dict): Default values for parameters not provided during interpolation.
            interpolator_function (function, optional): The interpolation function to use, defaults to RegularGridInterpolator.
            **interpolator_kwargs: Additional keyword arguments to pass to the interpolator function.
        """
        self.parameter_order = parameter_order
        self.interpolation_parameter_points = interpolation_parameter_points
        self.values = values
        self.default_values = default_values
        self.interpolator_kwargs = interpolator_kwargs
        self.interpolator = None  # Placeholder for the actual interpolator
        self.interpolator_function = interpolator_function

        self.create_interpolator(**{param_name: param_vals for param_name, param_vals in zip(self.parameter_order, self.interpolation_parameter_points)})

    def create_interpolator(self, **kwargs):
        """
        Creates the interpolator instance based on the provided grid points and values.

        This method organizes the provided grid points according to the parameter_order and initializes
        the interpolator with these grid points and the corresponding values.

        Args:
            **kwargs: Grid points for each parameter, provided as keyword arguments where the keys match the parameter names.
        """
        grid_points = [np.unique(kwargs[param]) for param in self.parameter_order if param in kwargs]

        if len(grid_points) != len(self.parameter_order):
            missing_params = set(self.parameter_order) - set(kwargs.keys())
            raise ValueError(f"Missing parameters for interpolation: {missing_params}")

        self.interpolator = self.interpolator_function(
            grid_points,
            self.values,
            **self.interpolator_kwargs
        )

    def __call__(self, **kwargs):
        """
        Performs interpolation for the given set of parameters. If any parameter is missing,
        its default value is used.

        Args:
            **kwargs: Parameters for which the interpolation should be performed, provided as keyword arguments.

        Returns:
            The result of the interpolation for the given parameters.
        """
        if self.interpolator is None:
            raise ValueError("Interpolator has not been created. Call create_interpolator first.")

        points = []
        for param in self.parameter_order:
            if param in kwargs:
                points.append(kwargs[param])
            elif param in self.default_values:
                points.append(self.default_values[param])
            else:
                raise ValueError(f"Missing value for {param} and no default value provided.")
        
        points = np.array(points).T  # Assumes points form a grid in N-dimensional space

        return self.interpolator(points)
