import numpy as np
from gammabayes.dark_matter.density_profiles import DM_Profiles
from gammabayes.dark_matter.models.Z2_ScalarSinglet import Z2_ScalarSinglet
from os import path
from gammabayes.dark_matter.density_profiles import DM_Profile
from gammabayes.likelihoods.irfs import irf_loglikelihood

import time



class combine_DM_models:

    def __init__(self, 
                 spectral_gen_class: Z2_ScalarSinglet, 
                 spatial_profile_class: DM_Profile, 
                 irf_class: irf_loglikelihood,
                 spectral_class_kwds: dict = {},
                 spatial_class_kwds: dict = {},
                 ):

        self.spectral_class_instance    = spectral_gen_class(**spectral_class_kwds)
        self.profile     = spatial_profile_class(**spatial_class_kwds)
        self.irf_class                  = irf_class

    def DM_signal_dist(self, energy: float | np.ndarray | list, 
                       longitude: float | np.ndarray | list,  
                       latitude: float | np.ndarray | list,  
                       spectral_parameters: dict = {'mass': 1.0, },
                       spatial_parameters: dict = {}, ) -> np.ndarray:

        energy, longitude, latitude = np.asarray(energy), np.asarray(longitude), np.asarray(latitude)

        spectral_parameters = {param_key: np.asarray(param_val) for param_key, param_val in spectral_parameters.items()}

        
        flatten_spectral_param_values = np.array([energy.flatten(), 
                                       *[param_val.flatten() for param_val in spectral_parameters.values()]])
        
        unique_spectral_param_values = np.unique(flatten_spectral_param_values, axis=1)

        logspectralvals = self.spectral_class_instance.logfunc(
             unique_spectral_param_values[0], 
             kwd_parameters = {
                  spectral_param_key : unique_spectral_param_values[1+val_idx] for val_idx, spectral_param_key in enumerate(spectral_parameters.keys())
                  },
                  )

        mask = np.all(unique_spectral_param_values[:, None, :] == flatten_spectral_param_values[:, :, None], axis=0)

        slices = np.where(mask, logspectralvals[None, :], 0.0)

        logspectralvals = np.sum(slices, axis=-1).reshape(energy.shape)

        ####################


        spatial_parameters = {param_key: np.asarray(param_val) for param_key, param_val in spatial_parameters.items()}

        flatten_spatial_param_vals = np.array([longitude.flatten(), latitude.flatten(), *[spatial_param_val.flatten() for spatial_param_val in spatial_parameters.values()]])
        
        unique_spatial_param_vals = np.unique(flatten_spatial_param_vals, axis=1)

        logspatialvals = self.profile.logdiffJ(
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
        log_aeffvals = self.irf_class.log_aeff(energy.flatten(), longitude.flatten(), latitude.flatten()).reshape(energy.shape)
    
        logpdfvalues = logspectralvals+logspatialvals+log_aeffvals

        
        return logpdfvalues
    
    def __call__(self, *args, **kwargs) -> np.ndarray:
         return self.DM_signal_dist(*args, **kwargs)
    

    def DM_signal_dist_mesh_efficient(self, 
                                      energy: float | np.ndarray | list,
                                      longitude: float | np.ndarray | list, 
                                      latitude: float | np.ndarray | list,
                                      spatial_parameters: dict = {}, 
                                      spectral_parameters: dict = {'mass': 1.0, }, 
                                      ) -> np.ndarray:
            
            num_spectral_params     = len(spectral_parameters)

            num_spatial_params      = len(spatial_parameters)

            # Just the total
            num_total_params = num_spectral_params + num_spatial_params + 3

            ####################

            logspectralvals     = self.spectral_class_instance.mesh_efficient_logfunc(energy, kwd_parameters=spectral_parameters)

            ####################

            logspatialvals      = self.profile.mesh_efficient_logdiffJ(longitude, latitude, kwd_parameters=spatial_parameters)

            ####################

            aeff_energy_mesh, aeff_lon_mesh, aeff_lat_mesh = np.meshgrid(energy, longitude, latitude, indexing='ij')

            # Flattening the meshes helps the IRFs evaluate
            log_aeffvals = self.irf_class.log_aeff(aeff_energy_mesh.flatten(), aeff_lon_mesh.flatten(), aeff_lat_mesh.flatten()).reshape(aeff_energy_mesh.shape)
            
            ####################

            # Convention is Energy, Lon, Lat, Mass, [Spectral_Params], [Spatial_Params]

            # Expanding along Lon, Lat, and DM_Spatial_Params dims
            expand_spectral_axes = list([1,2])+list(range(3+num_spectral_params, num_total_params))
            logpdfvalues = np.expand_dims(logspectralvals, 
                                          axis=expand_spectral_axes)
            

            # Expanding along Energy, Mass, and DM_Spectral_Params dims
            expand_spatial_axes = list([0])+list(range(3, 3+num_spectral_params))

            log_spatial_vals = np.expand_dims(logspatialvals, 
                                          axis=expand_spatial_axes)
            

            logpdfvalues = logpdfvalues + log_spatial_vals
            
            # Expanding along all the spectral and spatial parameters
            expand_aeff_axes = list(range(3, num_total_params))
            log_aeff_vals = np.expand_dims(log_aeffvals, 
                                          axis=expand_aeff_axes)
            

            logpdfvalues = logpdfvalues + log_aeff_vals

            return logpdfvalues

    
