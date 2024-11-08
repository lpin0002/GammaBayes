from astropy import units as u
import numpy as np
from gammabayes.priors.core.wrappers import _wrap_if_missing_keyword
from gammabayes import update_with_defaults

from .base_spatial_comp import BaseSpatial_PriorComp
from icecream import ic

class GaussianSpatial_PriorComp(BaseSpatial_PriorComp):

    __default_parameter_values = {'pos_lon_deg':0, 'pos_lat_deg':0, 
                                    "sigma_lon":0.1, "sigma_lat":0.1, "rho":0, 
                                    "normalisation":1.}

    @staticmethod
    def gaussian_logfunc(lon, lat, pos_lon_deg, pos_lat_deg, sigma_lon, sigma_lat, rho, normalisation=1., *args, **kwargs):

        longitude_factor = (lon.value-pos_lon_deg)/sigma_lon
        latitude_factor = (lat.value-pos_lat_deg)/sigma_lat

        prefactor = 1/(2*np.pi*sigma_lon*sigma_lat*np.sqrt(1-rho**2))
        exponent_prefactor = -1/(2*(1-rho**2))
        exponent_body = longitude_factor**2-2*rho*longitude_factor*latitude_factor+latitude_factor**2

        return np.log(normalisation*prefactor) + exponent_prefactor*exponent_body
    
    @staticmethod
    def gaussian_meshlogfunc(lon, lat, pos_lon_deg, pos_lat_deg, sigma_lon, sigma_lat, rho, normalisation=1., *args, **kwargs):

        lon_mesh, lat_mesh = np.meshgrid(lon, lat, indexing='ij')

        return GaussianSpatial_PriorComp.gaussian_logfunc(
            lon_mesh, lat_mesh, 
            pos_lon_deg, pos_lat_deg, 
            sigma_lon, sigma_lat, rho, 
            normalisation=1., *args, **kwargs)



    def __init__(self, default_parameter_values=None, *args, **kwargs):

        if default_parameter_values is None:
            default_parameter_values = {}


        for kwd, kwd_val in kwargs.items():
            default_parameter_values.setdefault(kwd, kwd_val)

        
        for kwd, kwd_val in GaussianSpatial_PriorComp.__default_parameter_values.items():
            default_parameter_values.setdefault(kwd, kwd_val)

        super().__init__(logfunc=self.gaussian_logfunc, 
                         default_parameter_values=default_parameter_values, *args, **kwargs)