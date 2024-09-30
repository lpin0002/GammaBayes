from astropy import units as u
import numpy as np
from gammabayes.priors.core.wrappers import _wrap_if_missing_keyword
from gammabayes import update_with_defaults

from .base_spatial_comp import BaseSpatial_PriorComp
from icecream import ic

class IsotropicSpatial_PriorComp(BaseSpatial_PriorComp):

    @staticmethod
    def iso_logfunc(lon, lat, *args, **kwargs):
        return lon.value*0
    
    @staticmethod
    def iso_meshlogfunc(lon, lat, *args, **kwargs):

        zero_output = np.zeros(shape=(len(lon), len(lat)))

        return zero_output
    
    def __init__(self, default_parameter_values=None, *args, **kwargs):

        super().__init__(logfunc=self.iso_logfunc, 
                         default_parameter_values=default_parameter_values, *args, **kwargs)