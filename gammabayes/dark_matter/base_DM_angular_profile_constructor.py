import numpy as np
from astropy import units as u
from gammapy.astro.darkmatter import (
    profiles,
    JFactory
)
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
from scipy import interpolate
import pandas as pd
from ..likelihoods.instrument_response_funcs import aefffunc
from os import path
darkmatter_dir = path.dirname(__file__)


# DM_dist(longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile())
class base_DM_angular_dist_constructor(object):
    
    def __init__(self, longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile(),):
        self.longitudeaxis = longitudeaxis
        self.latitudeaxis = latitudeaxis
        self.density_profile = density_profile
    

        # Adopt standard values used in HESS
        profiles.DMProfile.DISTANCE_GC = 8.5 * u.kpc
        profiles.DMProfile.LOCAL_DENSITY = 0.39 * u.Unit("GeV / cm3")

        self.density_profile.scale_to_local_density()

        self.central_position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
        self.geom = WcsGeom.create(skydir=self.central_position, 
                            binsz=(self.longitudeaxis[1]-self.longitudeaxis[0], self.latitudeaxis[1]-self.latitudeaxis[0]),
                            width=(self.longitudeaxis[-1]-self.longitudeaxis[0]+self.longitudeaxis[1]-self.longitudeaxis[0], self.latitudeaxis[-1]-self.latitudeaxis[0]+self.latitudeaxis[1]-self.latitudeaxis[0]),
                            frame="galactic")


        jfactory = JFactory(
            geom=self.geom, profile=self.density_profile, distance=profiles.DMProfile.DISTANCE_GC
        )
        self.diffjfact_array = (jfactory.compute_differential_jfactor().value).T

        self.diffJfactor_function = interpolate.RegularGridInterpolator((self.longitudeaxis, self.latitudeaxis), self.diffjfact_array, method='linear', bounds_error=False, fill_value=0)
    

    
    def log_func(self, longitude, latitude):
        
        return np.log(self.diffJfactor_function(longitude, latitude))
    
    