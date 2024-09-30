import numpy as np
from os import path
from gammabayes.priors.core.discrete_logprior import DiscreteLogPrior
from gammabayes.priors.core.source_flux_prior import SourceFluxDiscreteLogPrior
from gammabayes.core import GammaLogExposure, GammaBinning
import time

from icecream import ic

class TwoCompFluxPrior(SourceFluxDiscreteLogPrior):
    """
    A two-component prior model combining both spectral and spatial components adding exposure to source flux model.
    
    This class inherits from DiscreteLogPrior and combines a spectral and spatial model component 
    to calculate a log-prior distribution over a defined parameter space. Optionally, efficient 
    mesh-based log function computations for both spectral and spatial components can be specified.
    
    Args:
        spectral_class: A class representing the spectral model component. It must implement a method 
                        for calculating log values given parameters.
        
        spatial_class: A class representing the spatial model component. Similar to spectral_class, 
                       it must provide a method for calculating log values.
        
        spectral_mesh_efficient_logfunc (optional): A function for efficient log computation over a mesh grid for the spectral component. 
                                                     Defaults to None, indicating that the default method on spectral_class is used.
        
        spatial_mesh_efficient_logfunc (optional): Similar to spectral_mesh_efficient_logfunc, but for the spatial component.
                                                   Defaults to None.
        
        spectral_class_kwds (dict, optional): Keyword arguments to be passed to the spectral_class during initialization. Defaults to {}.
        
        spatial_class_kwds (dict, optional): Keyword arguments to be passed to the spatial_class during initialization. Defaults to {}.
        """



    def __init__(self, 
                 spectral_class, 
                 spatial_class, 
                 spectral_mesh_efficient_logfunc=None, 
                 spatial_mesh_efficient_logfunc=None, 
                 spectral_class_kwds: dict = {},
                 spatial_class_kwds: dict = {},
                 *args, **kwargs
                 ):
        """
        Initialize a two-component prior model combining both spectral and spatial components.
        
        This class inherits from DiscreteLogPrior and combines a spectral and spatial model component 
        to calculate a log-prior distribution over a defined parameter space. Optionally, efficient 
        mesh-based log function computations for both spectral and spatial components can be specified.
        
        Args:
            spectral_class: A class representing the spectral model component. It must implement a method 
                            for calculating log values given parameters.
            
            spatial_class: A class representing the spatial model component. Similar to spectral_class, 
                           it must provide a method for calculating log values.
                        
            spectral_mesh_efficient_logfunc (optional): A function for efficient log computation over a mesh grid for the spectral component. 
                                                         Defaults to None, indicating that the default method on spectral_class is used.
            
            spatial_mesh_efficient_logfunc (optional): Similar to spectral_mesh_efficient_logfunc, but for the spatial component.
                                                       Defaults to None.
            
            spectral_class_kwds (dict, optional): Keyword arguments to be passed to the spectral_class during initialization. Defaults to {}.
            
            spatial_class_kwds (dict, optional): Keyword arguments to be passed to the spatial_class during initialization. Defaults to {}.
            
            name (str, optional): A name for the two-component prior model. Defaults to 'UnknownTwoComp'.
        """

        self.spectral_comp    = spectral_class(**spectral_class_kwds)
        self.spatial_comp     = spatial_class(**spatial_class_kwds)

        if spectral_mesh_efficient_logfunc is None:
            spectral_mesh_efficient_exist = hasattr(self.spectral_comp, 'mesh_efficient_logfunc')
        else:
            spectral_mesh_efficient_exist = True

        if spatial_mesh_efficient_logfunc is None:
            spatial_mesh_efficient_exist = hasattr(self.spatial_comp, 'mesh_efficient_logfunc')
        else:
             spatial_mesh_efficient_exist = True
        
        self.mesh_efficient_exists = spatial_mesh_efficient_exist & spectral_mesh_efficient_exist


        if self.mesh_efficient_exists:
            super().__init__(
                log_flux_function = self.log_flux_function, 
                log_mesh_efficient_flux_func = self.log_mesh_efficient_flux_func,
                *args, **kwargs
                )
        else:
            super().__init__(
                log_flux_function = self.log_flux_function, 
                *args, **kwargs
                )
            

    def log_flux_function(self, energy: float | np.ndarray | list, 
                       lon: float | np.ndarray | list,  
                       lat: float | np.ndarray | list,  
                       spectral_parameters: dict = {},
                       spatial_parameters: dict = {}, 
                       ) -> np.ndarray:
        """
        Calculate the log prior distribution for given energy, longitude, and latitude values, 
        along with optional spectral and spatial parameters.
        
        This method computes the combined log prior values by integrating the spectral and spatial model components 
        and the instrument response function over the specified parameter space.
        
        Args:
            energy (float | np.ndarray | list): The energy values for which the log prior distribution is calculated.
            
            lon (float | np.ndarray | list): The longitude values for the spatial component.
            
            lat (float | np.ndarray | list): The latitude values for the spatial component.
            
            spectral_parameters (dict, optional): A dictionary of parameters specific to the spectral component. Defaults to {}.
            
            spatial_parameters (dict, optional): A dictionary of parameters specific to the spatial component. Defaults to {}.
            
        Returns:
            np.ndarray: The calculated log prior values as a numpy array.
        """


        spectral_axes = energy, *spectral_parameters.values()


        spectral_units = []
        for axis in spectral_axes:
            if hasattr(axis, 'unit'):
                spectral_units.append(axis.unit)
            else:
                spectral_units.append(1.)



        spatial_axes = lon, lat, *spatial_parameters.values()

        spatial_units = []
        for axis in spatial_axes:
            if hasattr(axis, 'unit'):
                spatial_units.append(axis.unit)
            else:
                spatial_units.append(1.)
        
            
        energy, lon, lat = np.asarray(energy), np.asarray(lon), np.asarray(lat)

        mesh_spectral_parameters = {param_key: np.asarray(param_val) for param_key, param_val in spectral_parameters.items()}
        
        flatten_spectral_param_values = np.array([energy.flatten(), 
                                       *[param_val.flatten() for param_val in mesh_spectral_parameters.values()]])
        
        #1 So what we're doing here is to pick out the unique combinations of the energy and spectral parameters
        unique_spectral_param_values = np.unique(flatten_spectral_param_values, axis=1)


        #2 We then only have to evaluate on these values for the spectral component
        logspectralvals = self.spectral_comp(
             unique_spectral_param_values[0]*spectral_units[0], 
             kwd_parameters = {
                  spectral_param_key : unique_spectral_param_values[1+val_idx]*spectral_units[1+val_idx] for val_idx, spectral_param_key in enumerate(mesh_spectral_parameters.keys())
                  },
                  )
        

        #3 We then create a mask to map where the unique parameter combination values are within the original parameter space
        mask = np.all(unique_spectral_param_values[:, None, :] == flatten_spectral_param_values[:, :, None], axis=0)


        #4 Then use the above map to assign the outputs to the relevant parameter combinations
        slices = np.where(mask, logspectralvals[None, :], 0.0)

        logspectralvals = np.sum(slices, axis=-1).reshape(energy.shape)

        ####################

        # The above unique value filtering is not applied here, as based on tests the performance benefits it typically small
            # If in the future many more spatial parameters are used we may include a switch to perform the same kind of
            # operations here
        spatial_parameters = {param_key: np.asarray(param_val) for param_key, param_val in spatial_parameters.items()}

        flatten_spatial_param_vals = np.array([lon.flatten(), lat.flatten(), *[spatial_param_val.flatten() for spatial_param_val in spatial_parameters.values()]])
        
        unique_spatial_param_vals = np.unique(flatten_spatial_param_vals, axis=1)

        logspatialvals = self.spatial_comp(
             unique_spatial_param_vals[0]*spatial_units[0], 
             unique_spatial_param_vals[1]*spatial_units[1], 
             kwd_parameters = {
                  param_key : unique_spatial_param_vals[2+val_idx]*spatial_units[2+val_idx] for val_idx, param_key in enumerate(spatial_parameters.keys())
                  }
                  )

        spatial_mask = np.all(unique_spatial_param_vals[:, None, :] == flatten_spatial_param_vals[:, :, None], axis=0)

        spatial_slices = np.where(spatial_mask, logspatialvals[None, :], 0.0)

        logspatialvals = np.sum(spatial_slices, axis=-1).reshape(energy.shape)

        ####################
    
        logpdfvalues = logspectralvals+logspatialvals

        
        return logpdfvalues
        

    def log_mesh_efficient_flux_func(self, 
                                energy: float | np.ndarray | list,
                                lon: float | np.ndarray | list, 
                                lat: float | np.ndarray | list,
                                spatial_parameters: dict = {}, 
                                spectral_parameters: dict = {}, 
                                ) -> np.ndarray:
        
        """
        An efficient version of `logfunction` that utilizes mesh grid computations for the spectral and spatial components.
        
        This method is designed to be used when `spectral_mesh_efficient_logfunc` and/or `spatial_mesh_efficient_logfunc` 
        are provided during class initialization, allowing for more efficient computation over grid meshes.
        
        Args:
            energy (float | np.ndarray | list): The energy values for which the log distribution is calculated.
            
            lon (float | np.ndarray | list): The longitude values for the spatial component.
            
            lat (float | np.ndarray | list): The latitude values for the spatial component.
            
            spatial_parameters (dict, optional): Parameters specific to the spatial component. Defaults to {}.
            
            spectral_parameters (dict, optional): Parameters specific to the spectral component. Defaults to {}.
            
        Returns:
            np.ndarray: The calculated log prior values as a numpy array, optimized for mesh grid computations.
        """

        num_spectral_params     = len(spectral_parameters)

        num_spatial_params      = len(spatial_parameters)

        # Just the total
        num_total_params = num_spectral_params + num_spatial_params + 3

        ####################

        logspectralvals     = self.spectral_comp.mesh_efficient_logfunc(energy, kwd_parameters=spectral_parameters)

        ####################

        logspatialvals      = self.spatial_comp.mesh_efficient_logfunc(lon, lat, kwd_parameters=spatial_parameters)

        ####################

        # Convention is Energy, Lon, Lat, Mass, [Spectral_Params], [Spatial_Params]

        # Expanding along Lon, Lat, and spatial param dims

        expand_spectral_axes = list([1,2])+list(range(3+num_spectral_params, num_total_params))
        logpdfvalues = np.expand_dims(logspectralvals, 
                                        axis=expand_spectral_axes)
        

        # Expanding along Energy, Mass, and Spectral_Params dims
        expand_spatial_axes = list([0])+list(range(3, 3+num_spectral_params))

        log_spatial_vals = np.expand_dims(logspatialvals, 
                                        axis=expand_spatial_axes)
        
        logpdfvalues = logpdfvalues + log_spatial_vals


        return np.squeeze(logpdfvalues)

    
    def log_mesh_integral_efficient_func(self, 
                                energy: float | np.ndarray | list,
                                lon: float | np.ndarray | list, 
                                lat: float | np.ndarray | list,
                                spatial_parameters: dict = {}, 
                                spectral_parameters: dict = {}, 
                                ) -> np.ndarray:
        
        """
        An efficient version of `logfunction` that utilizes mesh grid computations for the spectral and spatial components.
        
        This method is designed to be used when `spectral_mesh_efficient_logfunc` and/or `spatial_mesh_efficient_logfunc` 
        are provided during class initialization, allowing for more efficient computation over grid meshes.
        
        Args:
            energy (float | np.ndarray | list): The energy values for which the log distribution is calculated.
            
            lon (float | np.ndarray | list): The longitude values for the spatial component.
            
            lat (float | np.ndarray | list): The latitude values for the spatial component.
            
            spatial_parameters (dict, optional): Parameters specific to the spatial component. Defaults to {}.
            
            spectral_parameters (dict, optional): Parameters specific to the spectral component. Defaults to {}.
            
        Returns:
            np.ndarray: The calculated log prior values as a numpy array, optimized for mesh grid computations.
        """
            
        num_spectral_params     = len(spectral_parameters)

        num_spatial_params      = len(spatial_parameters)


        # Just the total
        num_total_params = num_spectral_params + num_spatial_params + 3

        ####################

        logspectralvals     = self.spectral_comp.mesh_integral_efficient_logfunc(energy, kwd_parameters=spectral_parameters)

        ####################

        logspatialvals      = self.spatial_comp.mesh_efficient_logfunc(lon, lat, kwd_parameters=spatial_parameters)


        ####################

        # Convention is Energy, Lon, Lat, Mass, [Spectral_Params], [Spatial_Params]

        # Expanding along Lon, Lat, and spatial param dims

        expand_spectral_axes = list([1,2])+list(range(3+num_spectral_params, num_total_params))
        logpdfvalues = np.expand_dims(logspectralvals, 
                                        axis=expand_spectral_axes)
        

        # Expanding along Energy, Mass, and Spectral_Params dims
        expand_spatial_axes = list([0])+list(range(3, 3+num_spectral_params))

        log_spatial_vals = np.expand_dims(logspatialvals, 
                                        axis=expand_spatial_axes)
        

        logpdfvalues = logpdfvalues + log_spatial_vals

        
        return np.squeeze(logpdfvalues)
