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
from gammabayes.likelihoods.irfs.gammapy_wrappers import aefffunc
from os import path
darkmatter_dir = path.dirname(path.dirname(__file__))
import time


from gammabayes.priors import discrete_logprior
from gammabayes.dark_matter.channel_spectra import single_channel_spectral_data_path

class SS_DM_Prior(discrete_logprior):
    
    def __init__(self, longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile(), ratios=False):
        self.longitudeaxis = longitudeaxis
        self.latitudeaxis = latitudeaxis
        self.density_profile = density_profile
        self.ratios = ratios
        logfunction = self.generate_log_function(axes, hyperparameter_axes, default_hyperparameter_values)
        super().__init__(name, inputunit, logfunction, axes, axes_names, hyperparameter_axes, hyperparameter_names, default_hyperparameter_values, logjacob)


        
    
    
        darkSUSY_to_PPPC_converter = {
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


        darkSUSY_BFs_cleaned = pd.read_csv(darkmatter_dir+'/darkSUSY_BFs/darkSUSY_BFs_cleaned.csv', delimiter=' ')

        darkSUSY_massvalues = darkSUSY_BFs_cleaned.iloc[:,1]/1e3

        darkSUSY_lambdavalues = darkSUSY_BFs_cleaned.iloc[:,2]

        sqrtchannelfuncdictionary = {}

       
        log10xvals = np.load(single_channel_spectral_data_path+f"/griddata/log10xvals_massenergy_diffflux_grid.npy")
        massvalues = np.load(single_channel_spectral_data_path+f"/griddata/massvals_massenergy_diffflux_grid.npy")

        for darkSUSYchannel in list(darkSUSY_to_PPPC_converter.keys()):
            try:
                gammapychannel = darkSUSY_to_PPPC_converter[darkSUSYchannel]
                
                tempspectragrid = np.load(single_channel_spectral_data_path+f"/griddata/channel={gammapychannel}_massenergy_diffflux_grid.npy")
                
                # Square root is to preserve positivity during interpolation
                sqrtchannelfuncdictionary[darkSUSYchannel] = interpolate.RegularGridInterpolator((np.log10(massvalues/1e3), log10xvals), np.sqrt(np.array(tempspectragrid)), 
                                                                                        method='cubic', bounds_error=False, fill_value=0)
            except:
                sqrtchannelfuncdictionary[darkSUSYchannel] = lambda logmass, log10x: log10x*0

        self.sqrtchannelfuncdictionary = sqrtchannelfuncdictionary
        
        darkSUSY_BFs_cleaned_vals = darkSUSY_BFs_cleaned.to_numpy()[:,3:]
        if self.ratios:
            darkSUSY_BFs_cleaned_vals = darkSUSY_BFs_cleaned_vals/np.sum(darkSUSY_BFs_cleaned_vals, axis=1)[:, np.newaxis]
            
        self.partial_sigmav_interpolator_dictionary = {channel: interpolate.LinearNDInterpolator((darkSUSY_massvalues, darkSUSY_lambdavalues),darkSUSY_BFs_cleaned_vals[:,idx]) for idx, channel in enumerate(list(darkSUSY_to_PPPC_converter.keys()))}
        
        self.profile = density_profile

        # Adopt standard values used in HESS
        profiles.DMProfile.DISTANCE_GC = 8.5 * u.kpc
        profiles.DMProfile.LOCAL_DENSITY = 0.39 * u.Unit("GeV / cm3")

        self.profile.scale_to_local_density()

        self.central_position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
        self.geom = WcsGeom.create(skydir=self.central_position, 
                            binsz=(self.longitudeaxis[1]-self.longitudeaxis[0], 
                                self.latitudeaxis[1]-self.latitudeaxis[0]),
                            width=(self.longitudeaxis[-1]-self.longitudeaxis[0]+self.longitudeaxis[1]-self.longitudeaxis[0], 
                                self.latitudeaxis[-1]-self.latitudeaxis[0]+self.latitudeaxis[1]-self.latitudeaxis[0]),
                            frame="galactic")


        jfactory = JFactory(
            geom=self.geom, profile=self.profile, distance=profiles.DMProfile.DISTANCE_GC
        )
        self.diffjfact_array = (jfactory.compute_differential_jfactor().value).T

        self.diffJfactor_function = interpolate.RegularGridInterpolator((self.longitudeaxis, self.latitudeaxis), self.diffjfact_array, method='linear', bounds_error=False, fill_value=0)


    def nontrivial_coupling(self, logmass, logenergy, coupling=0.1, 
                            partial_sigmav_interpolator_dictionary=None, sqrtchannelfuncdictionary=None):
        """Calculates Z_2 scalar singlet dark matter annihilation gamma-ray 
            spectra for a set of mass and coupling values.

        Args:
            logmass (float): Float value of log_10 of the dark matter mass in 
                TeV.

            logenergy (np.ndarray or float): Float values for log_10 gamma ray 
                energy values in TeV.

            coupling (float, optional): Value for the Higgs coupling. Defaults 
                to 0.1.

            partial_sigmav_interpolator_dictionary (dict, optional): A dictionary
                where the keys are the names of the dark matter annihilation final
                states as in DarkSUSY and the values being interpolation functions
                to calculate the partial annihilation cross-sections for the
                respective final states for a log_10 mass (TeV) and coupling 
                values. Defaults to None.

            channelfuncdictionary (dict, optional): A dictionary
                where the keys are the names of the dark matter annihilation final
                states as in DarkSUSY and the values being interpolation functions
                to calculate the spectral flux of gamma-rays for the respective final
                state. Defaults to None.

        Returns:
            np.ndarray: The total gamma-ray flux for the Z_2 Scalar Singlet 
                dark matter model
        """
        partial_sigmav_interpolator_dictionary = self.partial_sigmav_interpolator_dictionary
            
        sqrtchannelfuncdictionary = self.sqrtchannelfuncdictionary
        
        logspectra = -np.inf

        for channel in sqrtchannelfuncdictionary.keys():
            logspectra = np.logaddexp(logspectra, np.log(partial_sigmav_interpolator_dictionary[channel](10**logmass, coupling)*(sqrtchannelfuncdictionary[channel]((logmass, logenergy-logmass)))**2))
        
        return logspectra

    

    
    
    
    
    
    def generate_log_spectra(self):
        """A method that pumps out a function representing the natural log of 
            the flux of dark matter annihilation gamma rays for a given log 
            energy, sky position, log mass and higgs coupling value.
        """
        
        def DM_signal_dist(log10eval, lonval, latval, logmass, coupling=0.1):

            unique_log10eval = np.unique(log10eval.flatten())
            spectralvals = self.nontrivial_coupling(unique_log10eval*0+logmass.flatten()[0], unique_log10eval)
            mask = unique_log10eval[:, None] == log10eval.flatten()

            slices = np.where(mask, spectralvals[:, None], 0.0)

            spectralvals = np.sum(slices, axis=0).reshape(log10eval.shape)
            # except:
            #     spectralvals = np.squeeze(logspecfunc(logmass, log10eval))

            spatialvals = np.log(self.diffJfactor_function((lonval.flatten(), latval.flatten()))).reshape(log10eval.shape)


            log_aeffvals = np.log( 
                aefffunc(10**log10eval.flatten(), np.sqrt((lonval.flatten()**2)+(latval.flatten()**2)))
                ).reshape(log10eval.shape)
                    

            logpdfvalues = spectralvals+spatialvals+log_aeffvals
            
            return logpdfvalues
        
        return DM_signal_dist
    
    