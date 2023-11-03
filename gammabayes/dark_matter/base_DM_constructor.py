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
class DM_dist(object):
    
    def __init__(self, longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile(), 
                 ratios=False, partial_sigmav_interpolator_dictionary=None):
        self.longitudeaxis = longitudeaxis
        self.latitudeaxis = latitudeaxis
        self.density_profile = density_profile
        self.ratios = ratios
        self.partial_sigmav_interpolator_dictionary = partial_sigmav_interpolator_dictionary
    

    
        self.darkSUSY_to_PPPC_converter = {
            "nuenue":"nu_e",
            "e+e-": "e",
            "numunumu":"nu_mu",
            "mu+mu-":"mu",
            'nutaunutau':"nu_tau",
            "tau+tau-":"tau",
            "cc": "c",
            "bb": "b",
            "tt": "t",
            "W+W-": "W",
            "ZZ": "Z",
            "gg": "g",
            "gammagamma": "gamma",
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
        

        if self.partial_sigmav_interpolator_dictionary==None:
            darkSUSY_BFs_cleaned = pd.read_csv(darkmatter_dir+'/darkSUSY_BFs/darkSUSY_BFs_cleaned.csv', delimiter=' ')

            darkSUSY_massvalues = darkSUSY_BFs_cleaned.iloc[:,1]/1e3

            darkSUSY_lambdavalues = darkSUSY_BFs_cleaned.iloc[:,2]

            darkSUSY_BFs_cleaned_vals = darkSUSY_BFs_cleaned.to_numpy()[:,3:]
            if self.ratios:
                darkSUSY_BFs_cleaned_vals = darkSUSY_BFs_cleaned_vals/np.sum(darkSUSY_BFs_cleaned_vals, axis=1)[:, np.newaxis]
                
            self.partial_sigmav_interpolator_dictionary = {channel: interpolate.LinearNDInterpolator((darkSUSY_massvalues, darkSUSY_lambdavalues),darkSUSY_BFs_cleaned_vals[:,idx]) for idx, channel in enumerate(list(self.darkSUSY_to_PPPC_converter.keys()))}
        

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
    
    def partial_sigmav_interpolator_dictionary_format(self):
        return self.partial_sigmav_interpolator_dictionary

    def nontrivial_coupling(self, logmass, logenergy, coupling=0.1):
        """Calculates Z_2 scalar singlet dark matter annihilation gamma-ray 
            spectra for a set of mass and coupling values.

        Args:
            logmass (float): Float value of log_10 of the dark matter mass in 
                TeV.

            logenergy (np.ndarray or float): Float values for log_10 gamma ray 
                energy values in TeV.

            coupling (float, optional): Value for the Higgs coupling. Defaults 
                to 0.1.


        Returns:
            np.ndarray: The total gamma-ray flux for the dark matter model 
                specified through partial sigma v functions
        """

        
        logspectra = -np.inf

        for channel in self.channelfuncdictionary.keys():
            logspectra = np.logaddexp(logspectra, np.log(self.partial_sigmav_interpolator_dictionary[channel](10**logmass, coupling)*self.channelfuncdictionary[channel]((logmass, logenergy-logmass))))
        
        return logspectra

    

    
    
    
    
    
    def func_setup(self):
        """A method that pumps out a function representing the natural log of 
            the flux of dark matter annihilation gamma rays for a given log 
            energy, sky position, log mass and higgs coupling value.
        """
        
        def DM_signal_dist(log10eval, lonval, latval, logmass, coupling=0.1):
            # try:
            spectralvals = self.nontrivial_coupling(logmass.flatten(), log10eval.flatten()).reshape(log10eval.shape)
            # except:
            #     spectralvals = np.squeeze(logspecfunc(logmass, log10eval))
            
            spatialvals = np.log(self.diffJfactor_function((lonval.flatten(), latval.flatten()))).reshape(log10eval.shape)
            log_aeffvals = np.log( 
                aefffunc(10**log10eval.flatten(), np.sqrt((lonval.flatten()**2)+(latval.flatten()**2)))
                ).reshape(log10eval.shape)
                    
            logpdfvalues = spectralvals+spatialvals+log_aeffvals
            
            return logpdfvalues
        
        return DM_signal_dist
    
    