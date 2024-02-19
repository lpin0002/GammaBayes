import numpy as np
import astropy.units as u


import numpy as np
import astropy.units as u
from gammabayes.utils import logspace_riemann, haversine, update_with_defaults

import time




class DM_Profile(object):

    def scale_density_profile(self, density, distance, **kwargs):
        scale = (density / self.log_profile_func(distance, **kwargs))
        self.default_rho_s *= scale

    def __init__(self, log_profile_func: callable, 
                 LOCAL_DENSITY: float = 3.9*1e-4, 
                 dist_to_source: float = 8.33, 
                 annihilation: bool=1,
                 default_rho_s: float = 1., 
                 default_r_s: float = 28.4, 
                 angular_central_coords: np.ndarray = np.array([0,0]),
                 kwd_profile_default_vals: dict = {}):
        self.kpc_to_cm                  = 3.086e21
        self.log_profile_func           = log_profile_func
        self.LOCAL_DENSITY              = LOCAL_DENSITY
        self.DISTANCE                   = dist_to_source
        self.annihilation               = annihilation
        self.kwd_profile_default_vals   = kwd_profile_default_vals
        self.default_r_s                = default_r_s
        self.default_rho_s              = default_rho_s
        self.angular_central_coords     = angular_central_coords

        self.scale_density_profile(self.LOCAL_DENSITY, self.DISTANCE, **kwd_profile_default_vals)

    def __call__(self, *args, **kwargs) -> float | np.ndarray :
        return self.logdiffJ(*args, **kwargs)
    
    

    
    def _radius(self, t: float | np.ndarray, 
                angular_offset: float | np.ndarray, 
                distance: float | np.ndarray) -> float | np.ndarray :
        
        # converting angular_offset (in degrees) into radians
        t_mesh, offset_mesh = np.meshgrid(t, angular_offset*np.pi/180, indexing='ij')

        costheta = np.cos(offset_mesh)
        sintheta = np.sin(offset_mesh)
        inside = t_mesh**2*costheta**2*sintheta**2 + t_mesh**2*costheta**2 - 2*t_mesh*costheta + 1
        returnval = distance*np.sqrt(inside)

        return returnval


    def logdiffJ(self, longitude: float | np.ndarray, 
                 latitude: float | np.ndarray, 
              int_resolution: int = 1001, 
              integration_method: callable = logspace_riemann, 
              kwd_parameters = {}) -> float | np.ndarray :
        angular_offset = haversine(longitude, 
                                   latitude, 
                                   self.angular_central_coords[0], 
                                   self.angular_central_coords[1],)
        
        update_with_defaults(kwd_parameters, self.kwd_profile_default_vals)

        t = np.linspace(0, 6, int_resolution)
        logy= (1+self.annihilation)*self.log_profile_func(self._radius(t, angular_offset, self.DISTANCE),  **kwd_parameters)

        logintegral = integration_method(
            logy=logy,
            x=t, 
            axis=0)
        

        return logintegral+np.log(self.DISTANCE)+np.log(np.cos(angular_offset*np.pi/180)) + np.log(self.kpc_to_cm)
    

    def mesh_efficient_logfunc(self, longitude, latitude, kwd_parameters={}, *args, **kwargs) -> float | np.ndarray :

        parameter_meshes = np.meshgrid(longitude, latitude, *kwd_parameters.values(), indexing='ij')
        parameter_values_flattened_meshes = np.asarray([mesh.flatten() for mesh in parameter_meshes])

        return self(
            longitude=parameter_values_flattened_meshes[0], 
            latitude=parameter_values_flattened_meshes[1], 
            kwd_parameters = {param_key: parameter_values_flattened_meshes[2+idx] for idx, param_key in enumerate(kwd_parameters)},
            *args, 
            **kwargs
            ).reshape(parameter_meshes[0].shape)
