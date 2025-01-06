

try:
    from jax import numpy as np
    from jax import jit
    from jax.numpy import interp
    from gammabayes.utils.interpolation import JAX_RegularGrid_Linear_Interpolator as RegularGridInterpolator
    from jax.scipy.integrate import trapezoid as simps

except Exception as err:
    print(__file__, err)
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    from numpy import interp
    from scipy.integrate import simps
from numpy import ndarray
import numpy
from scipy.integrate import cumtrapz




import random, time, pickle
from tqdm import tqdm
from scipy.stats import norm as norm1d
import yaml, warnings, sys, os
from astropy import units as u
from astropy.units import Quantity
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


@jit
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculates the distance between two points on a sphere specified by longitude and latitude using the 
    Haversine formula and returns the separation in degrees.

    Args:
        lon1 (Quantity): Longitude of the first point.
        lat1 (Quantity): Latitude of the first point.
        lon2 (Quantity): Longitude of the second point.
        lat2 (Quantity): Latitude of the second point.

    Returns:
        Quantity: Angular separation between the two points in degrees.
    """
    # Convert degrees to radians

    lon1 = lon1*np.pi/180
    lat1 = lat1*np.pi/180

    lon2 = lon2*numpy.pi/180
    lat2 = lat2*numpy.pi/180


    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    angular_separation_rad = 2 * np.arcsin(np.sqrt(a))

    return angular_separation_rad*180/numpy.pi


def hdp_credible_interval_1d(y: ndarray, sigma: ndarray|list, x: ndarray) -> list[float, float]|list[float]:
    """
    Computes the highest density posterior credible interval for a 1D distribution.

    Args:
        y (ndarray): Probability density function values.
        sigma (ndarray | list): Standard deviation for credible interval calculation.
        x (ndarray): Points at which the pdf is evaluated.

    Returns:
        list[float, float] | list[float]: Credible interval or a single point for sigma=0.
    """
    y = y/simps(y=y, x=x)
    levels = np.linspace(0, y.max(),1000)

    areas = simps(y= (y>=levels[:, None])*y, x=x, axis=1)

    # interpolator = interp1d(y=levels, x=areas)
    if sigma!=0:
        prob_val = norm1d.cdf(sigma)-norm1d.cdf(-sigma)

        level = interp(prob_val, xp=areas, fp=levels)

        prob_array_indices = np.where(y>=level)

        return np.array([x[prob_array_indices[0][0]],x[prob_array_indices[0][-1]]])
    else:
        cdf = cumtrapz(y=y, x=x)

        probval = norm1d.cdf(sigma)

        probidx = np.argmax(cdf >= probval)

        return [x[probidx]]


def power_law(energy: float|Quantity, index: float, phi0: int|Quantity =1) -> float|Quantity:
    """
    Evaluates a power law function.

    Args:
        energy (float | Quantity): Energy values.
        index (float): Power law index.
        phi0 (int | Quantity, optional): Normalization constant. Defaults to 1.

    Returns:
        float | Quantity: Computed power law values.
    """
    warnings.warn("power_law will be deprecated after version 0.1.16. Please use the provided function in the prior.spectral components module.")
    return phi0*energy**(index)





def save_to_pickle(filename, object_to_save, write_mode='wb'):
    """
    Saves an object to a file using pickle with a specified write mode.

    Args:
        filename (str): The name of the file to save the object.
        object_to_save (object): The object to be saved.
        write_mode (str, optional): The mode in which the file is opened. Defaults to 'wb'.
    """
    with open(filename, write_mode) as file:
        pickle.dump(object_to_save, file)


def load_pickle(filename, load_mode='rb'):
    """
    Loads an object from a pickle file with a specified load mode.

    Args:
        filename (str): The name of the file to load the object from.
        load_mode (str, optional): The mode in which the file is opened. Defaults to 'rb'.

    Returns:
        object: The loaded object.
    """
    with open(filename, load_mode) as file:
        loaded_object = pickle.load(file)

    return loaded_object



def generate_unique_int_from_string(string):
    """
    Generates a pseudo-unique integer from a given string using a predefined list of prime numbers.

    Args:
        string (str): The input string.

    Returns:
        int: A unique integer generated from the string.
    """
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
    """
    Applies the stick-breaking process for Dirichlet distributions directly.

    Args:
        mixtures_fractions (list | tuple): Mixture fractions for the Dirichlet process.
        depth (int): The depth of the stick-breaking process.

    Returns:
        ndarray | float: Resulting mixture component.
    """
    axes = {}

    for prior_axes in axes_config.values():
        for axes_for_input_type in prior_axes.values():
            for axis in axes_for_input_type.items():
                axes.update({axis[0]:axis[1]})

    return axes


def apply_dirichlet_stick_breaking_direct(mixtures_fractions: list | tuple, 
                                            depth: int) -> ndarray | float:
    """_summary_

    Args:
        mixtures_fractions (list | tuple): _description_
        depth (int): _description_

    Returns:
        ndarray | float: _description_
    """

    dirichletmesh = 1

    for _dirichlet_i in range(depth):
        dirichletmesh*=(1-mixtures_fractions[_dirichlet_i])

    not_max_depth = depth!=len(mixtures_fractions)
    
    if not_max_depth:
        dirichletmesh*=mixtures_fractions[depth]

    return dirichletmesh



def bound_axis(axis: ndarray, 
                bound_type: str, 
                bound_radii: u.Quantity, 
                estimated_val: u.Quantity):
    """
    Bounds an axis within specified radii around an estimated value.

    Args:
        axis (ndarray): Axis values.
        bound_type (str): Type of bounding ('linear' or 'log10').
        bound_radii (float): Bounding radii.
        estimated_val (float): Estimated value around which to bound.

    Returns:
        tuple: Bounded axis and indices.
    """
    # axis_unit = axis.unit
    # axis = axis.value

    # bound_radii = bound_radii.to(axis_unit)

    # bound_radii = bound_radii.value


    # estimated_val = estimated_val.to(axis_unit)

    # estimated_val = estimated_val.value



    if bound_type=='linear':
        axis_indices = np.where(
        (axis>estimated_val-bound_radii) & (axis<estimated_val+bound_radii) )[0]

    elif bound_type=='log10':
        axis_indices = np.where(
        (np.log10(axis)>np.log10(estimated_val)-bound_radii) & (np.log10(axis)<np.log10(estimated_val)+bound_radii) )[0]
        
    temp_axis = axis[axis_indices]

    return temp_axis, axis_indices




def bin_centres_to_edges(axis: ndarray) -> ndarray:
    """
    Converts bin centers to bin edges for a given axis.

    Args:
        axis (ndarray): Bin centers.

    Returns:
        ndarray: Bin edges.
    """
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
