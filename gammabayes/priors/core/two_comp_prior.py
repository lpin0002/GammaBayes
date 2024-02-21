import numpy as np
from os import path
from gammabayes.priors.core.discrete_logprior import DiscreteLogPrior

import time



class TwoCompPrior(DiscreteLogPrior):

    def __init__(self, 
                 spectral_class, 
                 spatial_class, 
                 irf_loglike, #: IRF_LogLikelihood,
                 spectral_mesh_efficient_logfunc=None, 
                 spatial_mesh_efficient_logfunc=None, 
                 spectral_class_kwds: dict = {},
                 spatial_class_kwds: dict = {},
                 name='UnknownTwoComp',
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
            
            irf_loglike: An instance of a class representing the Instrument Response Function (IRF) log-likelihood.
                         This is used to calculate the log-likelihood of the data given the model parameters.
            
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
        self.irf_loglike       = irf_loglike

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
            super().__init__(logfunction=self.log_dist, 
                            log_mesh_efficient_func=self.log_dist_mesh_efficient,
                            name=name,
                            *args, **kwargs
                            )
        else:
            super().__init__(logfunction=self.log_dist, 
                            name=name,
                            *args, **kwargs
                            )

    def log_dist(self, energy: float | np.ndarray | list, 
                       longitude: float | np.ndarray | list,  
                       latitude: float | np.ndarray | list,  
                       spectral_parameters: dict = {},
                       spatial_parameters: dict = {}, ) -> np.ndarray:
        """
        Calculate the log prior distribution for given energy, longitude, and latitude values, 
        along with optional spectral and spatial parameters.
        
        This method computes the combined log prior values by integrating the spectral and spatial model components 
        and the instrument response function over the specified parameter space.
        
        Args:
            energy (float | np.ndarray | list): The energy values for which the log prior distribution is calculated.
            
            longitude (float | np.ndarray | list): The longitude values for the spatial component.
            
            latitude (float | np.ndarray | list): The latitude values for the spatial component.
            
            spectral_parameters (dict, optional): A dictionary of parameters specific to the spectral component. Defaults to {}.
            
            spatial_parameters (dict, optional): A dictionary of parameters specific to the spatial component. Defaults to {}.
            
        Returns:
            np.ndarray: The calculated log prior values as a numpy array.
        """

        energy, longitude, latitude = np.asarray(energy), np.asarray(longitude), np.asarray(latitude)

        spectral_parameters = {param_key: np.asarray(param_val) for param_key, param_val in spectral_parameters.items()}

        
        flatten_spectral_param_values = np.array([energy.flatten(), 
                                       *[param_val.flatten() for param_val in spectral_parameters.values()]])
        
        #1 So what we're doing here is to pick out the unique combinations of the energy and spectral parameters
        unique_spectral_param_values = np.unique(flatten_spectral_param_values, axis=1)

        #2 We then only have to evaluate on these values for the spectral component
        logspectralvals = self.spectral_comp(
             unique_spectral_param_values[0], 
             kwd_parameters = {
                  spectral_param_key : unique_spectral_param_values[1+val_idx] for val_idx, spectral_param_key in enumerate(spectral_parameters.keys())
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

        flatten_spatial_param_vals = np.array([longitude.flatten(), latitude.flatten(), *[spatial_param_val.flatten() for spatial_param_val in spatial_parameters.values()]])
        
        unique_spatial_param_vals = np.unique(flatten_spatial_param_vals, axis=1)

        logspatialvals = self.spatial_comp(
             unique_spatial_param_vals[0], 
             unique_spatial_param_vals[1], 
             kwd_parameters = {
                  param_key : unique_spatial_param_vals[2+val_idx] for val_idx, param_key in enumerate(spatial_parameters.keys())
                  }
                  )

        spatial_mask = np.all(unique_spatial_param_vals[:, None, :] == flatten_spatial_param_vals[:, :, None], axis=0)

        spatial_slices = np.where(spatial_mask, logspatialvals[None, :], 0.0)

        logspatialvals = np.sum(spatial_slices, axis=-1).reshape(energy.shape)

        ####################
        log_aeffvals = self.irf_loglike.log_aeff(energy.flatten(), longitude.flatten(), latitude.flatten()).reshape(energy.shape)
    
        logpdfvalues = logspectralvals+logspatialvals+log_aeffvals

        
        return logpdfvalues
    
    def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        Enable the class instance to be called as a function, directly invoking the `log_dist` method.
        
        This allows for a more intuitive use of the class instance for calculating log distributions.
        
        Returns:
            np.ndarray: The log prior values calculated by `log_dist`.
        """
        
        return self.log_dist(*args, **kwargs)
    

    def log_dist_mesh_efficient(self, 
                                energy: float | np.ndarray | list,
                                longitude: float | np.ndarray | list, 
                                latitude: float | np.ndarray | list,
                                spatial_parameters: dict = {}, 
                                spectral_parameters: dict = {}, 
                                ) -> np.ndarray:
        
        """
        An efficient version of `log_dist` that utilizes mesh grid computations for the spectral and spatial components.
        
        This method is designed to be used when `spectral_mesh_efficient_logfunc` and/or `spatial_mesh_efficient_logfunc` 
        are provided during class initialization, allowing for more efficient computation over grid meshes.
        
        Args:
            energy (float | np.ndarray | list): The energy values for which the log distribution is calculated.
            
            longitude (float | np.ndarray | list): The longitude values for the spatial component.
            
            latitude (float | np.ndarray | list): The latitude values for the spatial component.
            
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

        logspatialvals      = self.spatial_comp.mesh_efficient_logfunc(longitude, latitude, kwd_parameters=spatial_parameters)

        ####################

        aeff_energy_mesh, aeff_lon_mesh, aeff_lat_mesh = np.meshgrid(energy, longitude, latitude, indexing='ij')

        # Flattening the meshes helps the IRFs evaluate
        log_aeffvals = self.irf_loglike.log_aeff(aeff_energy_mesh.flatten(), aeff_lon_mesh.flatten(), aeff_lat_mesh.flatten()).reshape(aeff_energy_mesh.shape)
        
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
        
        # Expanding along all the spectral and spatial parameters
        expand_aeff_axes = list(range(3, num_total_params))
        log_aeff_vals = np.expand_dims(log_aeffvals, 
                                        axis=expand_aeff_axes)
        

        logpdfvalues = logpdfvalues + log_aeff_vals






        return np.squeeze(logpdfvalues)

    
