from scipy.interpolate import RegularGridInterpolator
from numpy import ndarray
import copy
from icecream import ic

try:
    from jax.nn import logsumexp as logsumexp
    from jax.scipy.interpolate import RegularGridInterpolator as JAX_RegularGrid_Linear_Interpolator
    from jax import vmap, grad
    import jax.numpy as np
    import jax
    
except Exception as err:
    print(__file__, err)
    import numpy as np
    
import numpy


    


class EnergySpatialTemplateInterpolator:

    def __init__(self, binning_geometry, data:ndarray, 
                 spectral_parameter_axes:dict = None, spatial_parameter_axes:dict = None, 
                 axis_index_to_parameter_map:dict = None,
                 interpolation_method:str = 'linear',
                 bounds_error:bool = False, fill_value:float = 0,
                 general_func_interpolator=JAX_RegularGrid_Linear_Interpolator):
        
        if spectral_parameter_axes is None:
            spectral_parameter_axes = {}

        if spatial_parameter_axes is None:
            spatial_parameter_axes = {}

        self.binning_geometry = binning_geometry
        self.__data = data

        self.spectral_parameter_axes        = copy.deepcopy(spectral_parameter_axes)
        self.spatial_parameter_axes         = copy.deepcopy(spatial_parameter_axes)

        self.interpolation_method   = interpolation_method
        self.bounds_error           = bounds_error
        self.fill_value             = fill_value

        self.combined_parameter_axes = copy.deepcopy(self.spectral_parameter_axes)

        self.combined_parameter_axes.update(self.spatial_parameter_axes)


        if axis_index_to_parameter_map is None:
            axis_index_to_parameter_map = {parameter_index:parameter_key for parameter_index, parameter_key in enumerate(self.combined_parameter_axes.keys())}


        self.axis_index_to_parameter_map    = axis_index_to_parameter_map


        __axis_index_to_parameter_axis = {axis_index:self.combined_parameter_axes[parmeter_axis_key] for axis_index, parmeter_axis_key in axis_index_to_parameter_map.items()}
        self.__parameter_keys_to_axis_index = {parmeter_axis_key:axis_index for axis_index, parmeter_axis_key in axis_index_to_parameter_map.items()}


        self.__sorted_axis_index_to_parameter_axis = {key: val for key, val in sorted(__axis_index_to_parameter_axis.items(), key=lambda item: item[0])}


        self.__sorted_parameter_axes = [axis for axis in self.__sorted_axis_index_to_parameter_axis.values()]


        self.__axes_in_order = [axis for axis in binning_geometry.axes] + self.__sorted_parameter_axes
        self.__axes_in_order[0] = np.log10(self.__axes_in_order[0])
        self.general_func_interpolator = general_func_interpolator

        self.__regular_grid_interpolator = self.general_func_interpolator(
            self.__axes_in_order, values=self.__data, method=interpolation_method,
            bounds_error=self.bounds_error, fill_value=self.fill_value
        )


    def __call__(self, energy:float, lon:float, lat:float, spectral_parameters=None, spatial_parameters=None, *args, **kwargs):

        if spectral_parameters is None:
            spectral_parameters = {}
        if spatial_parameters is None:
            spatial_parameters = {}

        input_energy = energy.flatten()
        input_lon = lon.flatten()
        input_lat = lat.flatten()

        parameter_input_dict = copy.deepcopy(spectral_parameters)
        parameter_input_dict.update(spatial_parameters)

        axis_index_to_parameter_input = {self.__parameter_keys_to_axis_index[parameter_key]: parameter_value for parameter_key, parameter_value in parameter_input_dict.items()}

        sorted_parameter_tuple_dict = sorted(axis_index_to_parameter_input.items(), key = lambda item: item[0])

        sorted_parameter_inputs = [parameter_input for parameter_axis_index, parameter_input in sorted_parameter_tuple_dict]


        interpolator_input = np.vstack([
            np.log10(input_energy), 
            input_lon, 
            input_lat, 
            *sorted_parameter_inputs
        ]).T

        print(interpolator_input.shape)

        return np.log(self.__regular_grid_interpolator(interpolator_input).reshape(energy.shape))
