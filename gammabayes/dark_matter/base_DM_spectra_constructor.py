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
class base_DM_spectra_constructor(object):
    
    def __init__(self, ratios=False, partial_sigmav_interpolator_dictionary=None, default_channel="W+W-"):
        self.ratios = ratios
        self.partial_sigmav_interpolator_dictionary = partial_sigmav_interpolator_dictionary
        self.default_channel = default_channel
    
        self.darkSUSY_to_PPPC_converter = {
            "nuenue":"nu_e",
            "e+e-": "e",
            "numunumu":"nu_mu",
            "mu+mu-":"mu",
            'nutaunutau':"nu_tau",
            "tau+tau-":"tau",
            "uu":"u",
            "dd":"d",
            "cc": "c",
            "ss":"s",
            "tt": "t",
            "bb": "b",
            "gammagamma": "gamma",
            "W+W-": "W",
            "ZZ": "Z",
            "gg": "g",
            "HH": "h",
        } 



        channelfuncdictionary = {}

       
        log10xvals = np.load(darkmatter_dir+f"/griddata/log10xvals_massenergy_diffflux_grid.npy")
        massvalues = np.load(darkmatter_dir+f"/griddata/massvals_massenergy_diffflux_grid.npy")

        for darkSUSYchannel in list(self.darkSUSY_to_PPPC_converter.keys()):
            try:
                gammapychannel = self.darkSUSY_to_PPPC_converter[darkSUSYchannel]
                
                tempspectragrid = np.load(darkmatter_dir+f"/griddata/channel={gammapychannel}_massenergy_diffflux_grid.npy")
                
                channelfuncdictionary[darkSUSYchannel] = interpolate.RegularGridInterpolator((np.log10(massvalues/1e3), log10xvals), np.array(tempspectragrid), 
                                                                                        method='cubic', bounds_error=False, fill_value=1e-3000)
            except:
                channelfuncdictionary[darkSUSYchannel] = lambda input_tuple: input_tuple[1]*0

        self.channelfuncdictionary = channelfuncdictionary
        
        # If no dictioanry for the partial sigmav interpolators is given then single channel is presumed, 
        #   the default of which is the W+W- channe
        if self.partial_sigmav_interpolator_dictionary==None:
                
            self.partial_sigmav_interpolator_dictionary = {}
            for darkSUSYchannel in list(self.darkSUSY_to_PPPC_converter.keys()):
                if darkSUSYchannel==self.default_channel:
                    self.partial_sigmav_interpolator_dictionary[darkSUSYchannel]  = lambda inputs: inputs[0]*1
                else:
                    self.partial_sigmav_interpolator_dictionary[darkSUSYchannel]  = lambda inputs: inputs[0]*0
        

    
    def partial_sigmav_interpolator_dictionary_format(self):
        return self.partial_sigmav_interpolator_dictionary

    def log_func(self, mass, energy, non_mass_sigmav_args=None):
        """Calculates W+W- dark matter annihilation gamma-ray 
            spectra for a set of mass values.

        Args:
            mass (np.ndarray or float): Float value of the dark matter mass in TeV.

            energy (np.ndarray or float): Float values for gamma ray 
                energy values in TeV.
                
            non_mass_sigmav_args (np.ndarray or float): Arguments for the partial sigmav 
                function given to the class that are not mass

        Returns:
            np.ndarray: The total gamma-ray flux for dark matter 
                W+W- channel annihilation 
                specified through partial sigma v functions
        """

        # Starting off with no flux and then adding the flux from each channel
        logspectra = -np.inf

        for channel in self.channelfuncdictionary.keys():
            logspectra = np.logaddexp(logspectra, 
                            np.log(self.partial_sigmav_interpolator_dictionary[channel](mass, **non_mass_sigmav_args)*self.channelfuncdictionary[channel]((np.log10(mass), np.log10(energy)-np.log10(mass)))))
        
        return logspectra
