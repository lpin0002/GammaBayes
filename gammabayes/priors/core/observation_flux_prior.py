from .discrete_logprior import DiscreteLogPrior



try:
    from jax.nn import logsumexp
except:
    from scipy.special import logsumexp

try:
    from jax import numpy as np
except:
    import numpy as np


from gammabayes.samplers import  integral_inverse_transform_sampler
from gammabayes.utils import iterate_logspace_integration, construct_log_dx_mesh
from gammabayes import update_with_defaults, GammaObs, GammaBinning, GammaLogExposure

import matplotlib.pyplot as plt
import warnings, logging
import h5py, pickle

# TODO: #TechDebt
class ObsFluxDiscreteLogPrior(DiscreteLogPrior):


    def __init__(self,
                 logfunction: callable=None, 
                 log_mesh_efficient_func: callable = None,
                 binning_geometry: GammaBinning = None,
                 default_spectral_parameters: dict = None,  
                 default_spatial_parameters: dict = None,  
                 irf_loglike=None,
                 live_times=None,
                 pointing_dirs=np.array([0., 0.]),
                 *args, 
                 **kwargs,
                 ):
        if default_spectral_parameters is None:
            default_spectral_parameters = {}
        if default_spatial_parameters is None:
            default_spatial_parameters = {}

        self.pointing_dirs = pointing_dirs
        self.live_times = live_times

        self._obs_logfunction = logfunction
        self._obs_log_mesh_efficient_func = log_mesh_efficient_func

        if log_mesh_efficient_func is None:
            self.log_mesh_efficient_func_input = None
        else:
            self.log_mesh_efficient_func_input = self.obs_log_mesh_efficient_func
        

        
        super().__init__(
                 logfunction=self.obs_logfunction, 
                 log_mesh_efficient_func=self.log_mesh_efficient_func_input,
                 binning_geometry=binning_geometry,
                 default_spectral_parameters=default_spectral_parameters,  
                 default_spatial_parameters=default_spatial_parameters,  
                 irf_loglike=irf_loglike,
                 *args,
                 **kwargs)
        

        # self.log_exposure_map basically is just for possibly complicated observation time maps
        self.log_exposure_map = GammaLogExposure(binning_geometry=self.binning_geometry, 
                                                 irfs=self.irf_loglike,
                                                 live_times=live_times,
                                                 pointing_dirs=pointing_dirs)

        

    def obs_logfunction(self, energy, lon, lat, 
                    log_exposure_map:GammaLogExposure=None, pointing_dirs = None, live_times = None, *args, **kwargs):
        if pointing_dirs is None:
            pointing_dirs = self.pointing_dirs
        if live_times is None:
            live_times = self.live_times
        
        
        
        if log_exposure_map is None:
            log_exposure_map = self.log_exposure_map
            log_exposure_map.live_times = live_times
            


        return self._obs_logfunction(energy, lon, lat, pointing_dir=pointing_dirs, *args, **kwargs) \
            + log_exposure_map(energy, lon, lat, pointing_dirs=pointing_dirs, live_times=live_times)

    def obs_log_mesh_efficient_func(self, energy, lon, lat, 
                                 log_exposure_map:GammaLogExposure = None, pointing_dirs = None, observation_time=None, *args, **kwargs):

        if log_exposure_map is None:
            log_exposure_map = self.log_exposure_map

        if pointing_dirs is None:
            pointing_dirs = self.pointing_dirs


        energy_mesh, lon_mesh, lat_mesh = np.meshgrid(energy, lon, lat, indexing='ij')

        # Not a fan of hitting maximum recursion depth? Me neither funnily enough
        return self._obs_log_mesh_efficient_func(energy, lon, lat, pointing_dir=pointing_dirs, *args, **kwargs) \
            + log_exposure_map(energy_mesh.flatten(), lon_mesh.flatten(), lat_mesh.flatten(), pointing_dir=pointing_dirs, observation_time=observation_time).reshape(energy_mesh.shape)

    
            