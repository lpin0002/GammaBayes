import numpy as np
from astropy import units as u
from gammapy.astro.darkmatter import JFactory, profiles

from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
from scipy import interpolate
import pandas as pd
from gammabayes.likelihoods.irfs import log_aeff
from os import path
from gammabayes.dark_matter.density_profiles import check_profile_module, DMProfiles
ScalarSinglet_Folder_Path = path.dirname(__file__)

from gammabayes.dark_matter.channel_spectra import single_channel_spectral_data_path
import time

# SS_DM_dist(longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile())
class SS_DM_dist(object):
    
    def __init__(self, longitudeaxis, latitudeaxis, density_profile=DMProfiles.Einasto, ratios=True):
        self.longitudeaxis = longitudeaxis
        self.latitudeaxis = latitudeaxis
        self.density_profile = density_profile
        self.ratios = ratios
        """Initialise an SS_DM_dist class instance.

        Args:
            longitudeaxis (np.ndarray): Array of the galactic longitude values 
                to sample for the calculation of the different J-factor

            latitudeaxis (np.ndarray): Array of the galactic latitude values 
                to sample for the calculation of the different J-factor

            density_profile (_type_, optional): The density profile to be used 
                for the calculation of the differential J-factor. Must be of
                the same type as the profile contained in the 
                gamma.astro.darkmatter.profiles module, attribute of the dark_matter.density_profiles.DMProfiles class,
                or string representing profile

                Defaults to profiles.EinastoProfile().

            ratios (bool, optional): A bool representing whether one wants to use the input differential cross-sections
                or the annihilation __ratios__. Defaults to False.
        """
        
    
        darkSUSY_to_PPPC_converter = {
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



        darkSUSY_BFs_cleaned = pd.read_csv(ScalarSinglet_Folder_Path+'/darkSUSY_BFs/darkSUSY_BFs_cleaned.csv', delimiter=' ')

        darkSUSY_massvalues = darkSUSY_BFs_cleaned.iloc[:,1]/1e3

        darkSUSY_lambdavalues = darkSUSY_BFs_cleaned.iloc[:,2]

        sqrtchannelfuncdictionary = {}

       
        log10xvals = np.load(single_channel_spectral_data_path+f"/griddata/log10xvals_massenergy_diffflux_grid.npy")
        massvalues = np.load(single_channel_spectral_data_path+f"/griddata/massvals_massenergy_diffflux_grid.npy")

        for darkSUSYchannel in list(darkSUSY_to_PPPC_converter.keys()):
            try:
                gammapychannel = darkSUSY_to_PPPC_converter[darkSUSYchannel]
                
                tempspectragrid = np.load(
                    single_channel_spectral_data_path+f"/griddata/channel={gammapychannel}_massenergy_diffflux_grid.npy")
                
                # Square root is to preserve positivity during interpolation
                sqrtchannelfuncdictionary[darkSUSYchannel] = interpolate.RegularGridInterpolator(
                    (np.log10(massvalues/1e3), log10xvals), 
                    np.sqrt(np.array(tempspectragrid)), 
                    method='cubic', bounds_error=False, fill_value=0)
            except:
                sqrtchannelfuncdictionary[darkSUSYchannel] = lambda inputs: inputs[0]*0 # inputs should be a tuple or list of log_10(mass) in TeV and log_10(x)

        self.sqrtchannelfuncdictionary = sqrtchannelfuncdictionary
        
        darkSUSY_BFs_cleaned_vals = darkSUSY_BFs_cleaned.to_numpy()[:,3:]
        if self.ratios:
            darkSUSY_BFs_cleaned_vals = darkSUSY_BFs_cleaned_vals/np.sum(darkSUSY_BFs_cleaned_vals, axis=1)[:, np.newaxis]
            
        self.partial_sigmav_interpolator_dictionary = {
            channel: interpolate.LinearNDInterpolator(
                (darkSUSY_massvalues, darkSUSY_lambdavalues),
                darkSUSY_BFs_cleaned_vals[:,idx]) for idx, channel in enumerate(list(darkSUSY_to_PPPC_converter.keys())
                )
                }

        density_profile = check_profile_module(density_profile)
        
        self.profile = density_profile

        # Adopt standard values used in HESS
        profiles.DISTANCE_GC = 8.5 * u.kpc
        profiles.LOCAL_DENSITY = 0.39 * u.Unit("GeV / cm3")

        self.profile.scale_to_local_density()

        self.central_position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")

        # Presuming even spacings
        self.geom = WcsGeom.create(skydir=self.central_position, 
                            binsz=(np.diff(self.longitudeaxis)[0], np.diff(self.latitudeaxis)[0]),
                            width=(np.ptp(self.longitudeaxis)+np.diff(self.longitudeaxis)[0], 
                                np.ptp(self.latitudeaxis)+np.diff(self.latitudeaxis)[0]),
                            frame="galactic")


        jfactory = JFactory(
            geom=self.geom, profile=self.profile, distance=profiles.DMProfile.DISTANCE_GC
        )
        self.diffjfact_array = jfactory.compute_differential_jfactor().to(u.TeV**2/(u.cm**5*u.sr))
        self.diffjfact_array = (self.diffjfact_array.value).T
        self.diffJfactor_function = interpolate.RegularGridInterpolator(
            (self.longitudeaxis, self.latitudeaxis), 
            self.diffjfact_array, 
            method='linear', bounds_error=False, fill_value=0)


    def nontrivial_coupling(self, mass, energy, coupling=0.1):
        """Calculates Z_2 scalar singlet dark matter annihilation gamma-ray 
            spectra for a set of mass and coupling values.

        Args:
            mass (float): Float value of the dark matter mass in TeV.

            energy (np.ndarray or float): Float values for gamma ray energy 
                values in TeV.

            coupling (float, optional): Value for the Higgs coupling. Defaults 
                to 0.1.

        Returns:
            np.ndarray: The total gamma-ray flux for the Z_2 Scalar Singlet 
                dark matter model
        """
        partial_sigmav_interpolator_dictionary = self.partial_sigmav_interpolator_dictionary
            
        sqrtchannelfuncdictionary = self.sqrtchannelfuncdictionary
        
        logspectra = -np.inf

        for channel in sqrtchannelfuncdictionary.keys():
            logspectra = np.logaddexp(
                logspectra, 
                np.log(
                    partial_sigmav_interpolator_dictionary[channel](mass, coupling)\
                        *(sqrtchannelfuncdictionary[channel]((np.log10(mass), 
                        np.log10(energy)-np.log10(mass))))**2))
        
        return logspectra

    

    
    
    
    
    
    def func_setup(self):
        """A method that pumps out a function representing the natural log of 
            the flux of dark matter annihilation gamma rays for a given log 
            energy, sky position, log mass and higgs coupling value.
        """
        
        def DM_signal_dist(energyval, lonval, latval, mass, coupling=0.1):

            unique_energyval = np.unique(energyval.flatten())
            spectralvals = self.nontrivial_coupling(unique_energyval*0+mass.flatten()[0], unique_energyval)
            mask = unique_energyval[:, None] == energyval.flatten()

            slices = np.where(mask, spectralvals[:, None], 0.0)

            spectralvals = np.sum(slices, axis=0).reshape(energyval.shape)

            spatialvals = np.log(
                self.diffJfactor_function((lonval.flatten(), latval.flatten()))
                ).reshape(energyval.shape)


            log_aeffvals = log_aeff(energyval.flatten(), lonval.flatten(), latval.flatten()).reshape(energyval.shape)
                    

            logpdfvalues = spectralvals+spatialvals+log_aeffvals
            
            return logpdfvalues
        
        return DM_signal_dist
    
    