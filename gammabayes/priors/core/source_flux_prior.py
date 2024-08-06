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


class SourceFluxDiscreteLogPrior(DiscreteLogPrior):


    def __init__(self, name: str='[None]', 
                 inputunits: str=None, 
                 log_flux_function: callable=None, 
                 log_mesh_efficient_flux_func: callable = None,
                 axes: tuple[np.ndarray] | None = None, 
                 binning_geometry: GammaBinning = None,
                 default_spectral_parameters: dict = {},  
                 default_spatial_parameters: dict = {},  
                 iterative_logspace_integrator: callable = iterate_logspace_integration,
                 irf_loglike=None,
                 log_exposure_map=None, 
                 pointing_dir:np.ndarray[u.Quantity]=np.array([0.,0.])*u.deg,
                 observation_times=None,
                 observation_times_unit=None,
                 log_scaling_factor=0.,
                 ):
                
        
        self.log_flux_function = log_flux_function
        self.log_mesh_efficient_flux_func = log_mesh_efficient_flux_func

        if not(self.log_mesh_efficient_flux_func is None):
            self.log_mesh_efficient_func = self._log_mesh_efficient_func
        else:
            self.log_mesh_efficient_func = None
        

        
        super().__init__(name=name, 
                 inputunits=inputunits, 
                 logfunction=self.logfunction, 
                 log_mesh_efficient_func=self.log_mesh_efficient_func,
                 axes=axes, 
                 binning_geometry=binning_geometry,
                 default_spectral_parameters=default_spectral_parameters,  
                 default_spatial_parameters=default_spatial_parameters,  
                 iterative_logspace_integrator=iterative_logspace_integrator,
                 irf_loglike=irf_loglike,
                 log_scaling_factor=log_scaling_factor,)
        
        self.log_exposure_map = GammaLogExposure(binning_geometry=self.binning_geometry, 
                                                 irfs=self.irf_loglike,
                                                 log_exposure_map=log_exposure_map, 
                                                 pointing_dir=pointing_dir, 
                                                 observation_times=observation_times,
                                                 observation_times_unit=observation_times_unit)

        

    def logfunction(self, energy, lon, lat, log_exposure_map:GammaLogExposure=None, *args, **kwargs):
        if log_exposure_map is None:
            log_exposure_map = self.log_exposure_map


        return self.log_flux_function(energy, lon, lat, *args, **kwargs) + log_exposure_map(energy, lon, lat)

    def _log_mesh_efficient_func(self, energy, lon, lat, log_exposure_map:GammaLogExposure = None, *args, **kwargs):

        if log_exposure_map is None:
            log_exposure_map = self.log_exposure_map


        energy_mesh, lon_mesh, lat_mesh = np.meshgrid(energy, lon, lat, indexing='ij')



        return self.log_mesh_efficient_flux_func(energy, lon, lat, *args, **kwargs) + log_exposure_map(energy_mesh.flatten(), lon_mesh.flatten(), lat_mesh.flatten()).reshape(energy_mesh.shape)

    
    def log_source_flux(self, energy, lon, lat, log_scaling_factor:float=None, *args, **kwargs):

        if log_scaling_factor is None:
            log_scaling_factor = self.log_scaling_factor

        return log_scaling_factor+self.log_flux_function(energy, lon, lat, *args, **kwargs)

    def log_source_flux_mesh_efficient(self, energy, lon, lat, log_scaling_factor:float=None, *args, **kwargs):

        if log_scaling_factor is None:
            log_scaling_factor = self.log_scaling_factor

        return log_scaling_factor+self.log_mesh_efficient_flux_func(energy, lon, lat, *args, **kwargs) 
            