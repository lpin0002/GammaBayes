import numpy as np
from gammabayes.dark_matter.density_profiles import DM_Profiles
from gammabayes.dark_matter.models.Z2_ScalarSinglet import SS_Spectra
from os import path
from gammabayes.dark_matter.density_profiles import DM_Profile
from gammabayes.likelihoods.irfs import irf_loglikelihood

import time



class combine_DM_models:

    def __init__(self, 
                 spectral_gen_class: SS_Spectra, 
                 spatial_profile_class: DM_Profile, 
                 irf_class: irf_loglikelihood,
                 spectral_class_kwds: dict = {},
                 spatial_class_kwds: dict = {},
                 ) -> None:

        self.spectral_class_instance    = spectral_gen_class(**spectral_class_kwds)
        self.profile     = spatial_profile_class(**spatial_class_kwds)
        self.irf_class                  = irf_class

    def DM_signal_dist(self, energy: float | np.ndarray | list, 
                       lonval: float | np.ndarray | list,  
                       latval: float | np.ndarray | list,  
                       mass: float | np.ndarray | list,  
                       spectral_params: dict = {'coupling':0.1}) -> np.ndarray:

        mass = np.asarray(mass)
        spectral_params = {spectral_param_key: np.asarray(spectral_param_val) for spectral_param_key, spectral_param_val in spectral_params.items()}


        flatten_param_vals = np.array([energy.flatten(), mass.flatten(), *[spectral_param_val.flatten() for spectral_param_val in spectral_params.values()]])
        
        unique_param_vals = np.unique(flatten_param_vals, axis=1)

        logspectralvals = self.spectral_class_instance.spectral_gen(unique_param_vals[0], unique_param_vals[1], {spectral_param_key:unique_param_vals[2+val_idx] for val_idx, spectral_param_key in enumerate(spectral_params.keys())})

        mask = np.all(unique_param_vals[:, None, :] == flatten_param_vals[:, :, None], axis=0)

        slices = np.where(mask, logspectralvals[None, :], 0.0)

        logspectralvals = np.sum(slices, axis=-1).reshape(energy.shape)

        ####################

        flatten_spatial_param_vals = np.array([lonval.flatten(), latval.flatten(),])
        unique_spatial_param_vals = np.unique(flatten_spatial_param_vals, axis=1)

        logspatialvals = self.profile.logdiffJ(unique_spatial_param_vals)

        spatial_mask = np.all(unique_spatial_param_vals[:, None, :] == flatten_spatial_param_vals[:, :, None], axis=0)

        spatial_slices = np.where(spatial_mask, logspatialvals[None, :], 0.0)

        logspatialvals = np.sum(spatial_slices, axis=-1).reshape(energy.shape)

        ####################
        log_aeffvals = self.irf_class.log_aeff(energy.flatten(), lonval.flatten(), latval.flatten()).reshape(energy.shape)

    
        logpdfvalues = logspectralvals+logspatialvals+log_aeffvals

        
        return logpdfvalues
    
    def __call__(self, *args, **kwargs) -> np.ndarray:
         return self.DM_signal_dist(*args, **kwargs)
    

    def DM_signal_dist_mesh_efficient(self, 
                                      energy: float | np.ndarray | list,
                                      lonvals: float | np.ndarray | list, 
                                      latvals: float | np.ndarray | list,
                                      massvals: float | np.ndarray | list, 
                                      spatial_params: dict = {}, 
                                      dark_matter_params: dict = {'coupling':[0.1]}, 
                                      ) -> np.ndarray:
            
            # Must have mass
            num_spectral_params     = 1 + len(dark_matter_params)

            num_spatial_params      = len(spatial_params)

            # Just the total
            num_total_params = num_spectral_params + num_spatial_params + 3

            ####################

            logspectralvals     = self.spectral_class_instance.mesh_efficient_spectral_gen(energy, massvals, dark_matter_params)

            ####################

            logspatialvals      = self.profile.mesh_efficient_logdiffJ(lonvals, latvals)

            ####################

            aeff_energy_mesh, aeff_lon_mesh, aeff_lat_mesh = np.meshgrid(energy, lonvals, latvals, indexing='ij')

            log_aeffvals = self.irf_class.log_aeff(aeff_energy_mesh.flatten(), aeff_lon_mesh.flatten(), aeff_lat_mesh.flatten()).reshape(aeff_energy_mesh.shape)

            ####################

            # Convention is Energy, Lon, Lat, Mass, [DM_Spectral_Params], [DM_Spatial_Params]

            # Expanding along Lon, Lat, and DM_Spatial_Params dims
            logpdfvalues = np.expand_dims(logspectralvals, 
                                          axis=list([1,2])+list(range(3+num_spectral_params, num_total_params)))
            

            # Expanding along Energy, Mass, and DM_Spectral_Params dims
            expand_spatial_axes = list([0])+list(range(3, 3+num_spectral_params))


            log_spatial_vals = np.expand_dims(logspatialvals, 
                                          axis=expand_spatial_axes)
            
            logpdfvalues = logpdfvalues + log_spatial_vals
            
            # Expanding along mass, DM_Spectral_Params, and DM_Spatial_Params dims
            logpdfvalues += np.expand_dims(log_aeffvals, 
                                          axis=list(range(3, num_total_params)))

            return logpdfvalues

    
