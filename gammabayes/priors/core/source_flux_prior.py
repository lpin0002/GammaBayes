from .discrete_logprior import DiscreteLogPrior
from scipy.special import logsumexp
import numpy as np
from gammabayes.samplers import  integral_inverse_transform_sampler
from gammabayes.utils import iterate_logspace_integration, construct_log_dx_mesh
from gammabayes import update_with_defaults, GammaObs, GammaBinning, GammaLogExposure

from astropy import units as u
import matplotlib.pyplot as plt
import warnings, logging
import h5py, pickle
from icecream import ic

class SourceFluxDiscreteLogPrior(DiscreteLogPrior):


    def __init__(self,
                 log_flux_function: callable=None, 
                 log_mesh_efficient_flux_func: callable = None,
                 axes: tuple[np.ndarray] | None = None, 
                 binning_geometry: GammaBinning = None,
                 irf_loglike=None,
                 log_exposure_map=None, 
                 pointing_dir:np.ndarray[u.Quantity]=np.array([0.,0.])*u.deg,
                 observation_time=None,
                 observation_time_unit=None,
                 log_scaling_factor=0.,
                 *args,
                 **kwargs
                 ):
                
        
        self.log_flux_function = log_flux_function
        self.log_mesh_efficient_flux_func = log_mesh_efficient_flux_func

        if not(log_mesh_efficient_flux_func is None):
            self.log_mesh_efficient_func_input = self.unscaled_log_mesh_efficient_func
        else:
            self.log_mesh_efficient_func_input = None
        

        
        super().__init__(
                 logfunction=self.unscaled_logfunction, 
                 log_mesh_efficient_func=self.log_mesh_efficient_func_input,
                 axes=axes, 
                 binning_geometry=binning_geometry,
                 irf_loglike=irf_loglike,
                 log_scaling_factor=log_scaling_factor,
                 *args, 
                 **kwargs)
        
        self.log_exposure_map = GammaLogExposure(binning_geometry=self.binning_geometry, 
                                                 irfs=self.irf_loglike,
                                                 log_exposure_map=log_exposure_map, 
                                                 pointing_dir=pointing_dir, 
                                                 observation_time=observation_time,
                                                 observation_time_unit=observation_time_unit)

        

    def unscaled_logfunction(self, energy, lon, lat, log_exposure_map:GammaLogExposure=None, *args, **kwargs):
        if log_exposure_map is None:
            log_exposure_map = self.log_exposure_map


        return self.log_flux_function(energy, lon, lat, *args, **kwargs) + log_exposure_map(energy, lon, lat)

    def unscaled_log_mesh_efficient_func(self, energy, lon, lat, 
                                 log_exposure_map:GammaLogExposure = None, 
                                 spatial_parameters: dict = {}, 
                                spectral_parameters: dict = {}, 
                                *args, **kwargs):
        

        if log_exposure_map is None:
            log_exposure_map = self.log_exposure_map

        num_spectral_params     = len(spectral_parameters)

        num_spatial_params      = len(spatial_parameters)

        log_output_values = self.log_mesh_efficient_flux_func(energy, lon, lat, 
                                                              spectral_parameters=spectral_parameters, 
                                                              spatial_parameters=spatial_parameters, *args, **kwargs)

        # Just the total
        num_total_params = num_spectral_params + num_spatial_params + 3

        energy_mesh, lon_mesh, lat_mesh = self.binning_geometry.axes_mesh

        # Expanding along all the spectral and spatial parameters


        
        log_exposure_vals = log_exposure_map(energy_mesh.flatten(), lon_mesh.flatten(), lat_mesh.flatten()).reshape(energy_mesh.shape)

        # TODO: #TechDebt
        return (log_output_values.T+log_exposure_vals.T).T

    
    def log_source_flux(self, energy, lon, lat, log_scaling_factor:float=None, *args, **kwargs):

        if log_scaling_factor is None:
            log_scaling_factor = self.log_scaling_factor

        return log_scaling_factor+self.log_flux_function(energy, lon, lat, *args, **kwargs)

    def log_source_flux_mesh_efficient(self, energy, lon, lat, log_scaling_factor:float=None, *args, **kwargs):

        if log_scaling_factor is None:
            log_scaling_factor = self.log_scaling_factor

        return log_scaling_factor+self.log_mesh_efficient_flux_func(energy, lon, lat, *args, **kwargs) 
            