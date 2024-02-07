import numpy as np
from scipy import interpolate
import pandas as pd
from gammabayes.likelihoods.irfs import log_aeff
from gammabayes.dark_matter.density_profiles import DM_Profiles
from os import path
ScalarSinglet_Folder_Path = path.dirname(__file__)

from gammabayes.dark_matter.channel_spectra import single_channel_spectral_data_path
import time

# SS_DM_dist(longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile())
class SS_DM_dist(object):
    
    def __init__(self, longitudeaxis, latitudeaxis, density_profile=DM_Profiles.Einasto_Profile, density_kwargs={}, ratios=True):
        self.density_profile = density_profile
        self.density_kwargs     = density_kwargs
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

                Defaults to DM_Profiles.Einasto_Profile.

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
        
        self.profile = density_profile(**density_kwargs)


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
            
            flatten_param_vals = np.array([mass.flatten(), energyval.flatten(),])
            unique_param_vals = np.unique(flatten_param_vals, axis=1)

            logspectralvals = self.nontrivial_coupling(*unique_param_vals)

            mask = np.all(unique_param_vals[:, None, :] == flatten_param_vals[:, :, None], axis=0)

            slices = np.where(mask, logspectralvals[None, :], 0.0)

            logspectralvals = np.sum(slices, axis=-1).reshape(energyval.shape)

            ####################

            flatten_spatial_param_vals = np.array([lonval.flatten(), latval.flatten(),])
            unique_spatial_param_vals = np.unique(flatten_spatial_param_vals, axis=1)

            logspatialvals = self.profile.logdiffJ(unique_spatial_param_vals)

            spatial_mask = np.all(unique_spatial_param_vals[:, None, :] == flatten_spatial_param_vals[:, :, None], axis=0)

            spatial_slices = np.where(spatial_mask, logspatialvals[None, :], 0.0)

            logspatialvals = np.sum(spatial_slices, axis=-1).reshape(energyval.shape)

            ####################
            log_aeffvals = log_aeff(energyval.flatten(), lonval.flatten(), latval.flatten()).reshape(energyval.shape)

        
            logpdfvalues = logspectralvals+logspatialvals+log_aeffvals

            
            return logpdfvalues
        

        def DM_signal_dist_mesh_efficient(energyvals, lonvals, latvals, mass, coupling=0.1):

            
            energy_mesh, mass_mesh = np.meshgrid(energyvals, mass, indexing='ij')

            logspectralvals = self.nontrivial_coupling(mass_mesh.flatten(), energy_mesh.flatten()).reshape(energyvals.shape)


            ####################

            lon_mesh, lat_mesh = np.meshgrid(lonvals, latvals, indexing='ij')

            logspatialvals = self.profile.logdiffJ(np.array([lon_mesh.flatten(), lat_mesh.flatten()])).reshape(lon_mesh.shape)


            ####################

            aeff_energy_mesh, aeff_lon_mesh, aeff_lat_mesh = np.meshgrid(energyvals, lonvals, latvals, indexing='ij')

            log_aeffvals = log_aeff(aeff_energy_mesh.flatten(), aeff_lon_mesh.flatten(), aeff_lat_mesh.flatten()).reshape(aeff_energy_mesh.shape)

            logpdfvalues = logspectralvals[:, None, None] +logspatialvals[None, :, :]+log_aeffvals

            return logpdfvalues
        
        return DM_signal_dist, DM_signal_dist_mesh_efficient
    
    