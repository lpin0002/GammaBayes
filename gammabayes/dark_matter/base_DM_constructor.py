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
from .base_DM_angular_profile_constructor import base_DM_angular_dist_constructor
from .base_DM_spectra_constructor import base_DM_spectra_constructor
darkmatter_dir = path.dirname(__file__)


# DM_dist(longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile())
class construct_DM_dist(object):
    
    def __init__(self,  angular_dist_constructor = base_DM_angular_dist_constructor, 
                       spectral_dist_constructor = base_DM_spectra_constructor,
                longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile(), 
                 ratios=False, partial_sigmav_interpolator_dictionary=None):

        self.longitudeaxis      = longitudeaxis
        self.latitudeaxis       = latitudeaxis
        self.density_profile    = density_profile
        self.ratios = ratios
        self.partial_sigmav_interpolator_dictionary = partial_sigmav_interpolator_dictionary

        self.spectral_dist_constructor = spectral_dist_constructor(partial_sigmav_interpolator_dictionary=partial_sigmav_interpolator_dictionary, 
                                                                    ratios=ratios,)

        self.angular_dist_constructor = angular_dist_constructor(longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile())

        self.log_spectral_func       = self.spectral_dist_constructor.log_func()
        self.log_angular_func        = self.angular_dist_constructor.log_func()
    

    def construct_func(self):
        """A method that pumps out a function representing the natural log of 
            the flux of dark matter annihilation gamma rays for a given log 
            energy, sky position, log mass and higgs coupling value.
        """
        
        def DM_dist(energy_val, lonval, latval, mass, partial_sigmav_args):

            spectralvals = self.nontrivial_coupling(
                mass.flatten(), 
                energy_val.flatten(), 
                non_mass_sigmav_args=partial_sigmav_args).reshape(energy_val.shape)
            
            spatialvals = self.log_spectral_func(
                (lonval.flatten(), latval.flatten())).reshape(energy_val.shape)

            log_aeffvals = log_aeff(
                energy_val.flatten(), lonval.flatten(), latval.flatten()).reshape(energyval.shape)
                    
            logpdfvalues = spectralvals+spatialvals+log_aeffvals
            
            return logpdfvalues
        
        return DM_dist
    
    