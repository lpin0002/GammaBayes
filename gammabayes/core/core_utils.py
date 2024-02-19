import numpy as np
from scipy.interpolate import RegularGridInterpolator


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
