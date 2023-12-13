import numpy as np
from scipy import interpolate
import pandas as pd
from gammabayes.dark_matter.density_profiles import DM_Profiles
from os import path
ScalarSinglet_Folder_Path = path.dirname(__file__)

from gammabayes.dark_matter.channel_spectra import single_channel_spectral_data_path
import time

# SS_DM_dist(longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile())
class SS_Spectra(object):
    
    def __init__(self, ratios: bool = True):

        self.ratios = ratios
    
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
        


    def point_spectral_gen(self, energy: float | np.ndarray | list, 
                           mass: float | np.ndarray | list, 
                           coupling: float | np.ndarray | list = 0.1) -> np.ndarray | float:
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

    

    
        
    def spectral_gen(self, 
                     energy: list | np.ndarray | float, 
                     mass: list | np.ndarray | float, 
                     theta_params: dict = {'coupling': 0.1}) -> np.ndarray | float:
        energy = np.asarray(energy)
        mass = np.asarray(mass)

        for key, val in theta_params.items():
            asarray_param = np.asarray(val) 
            theta_params[key] = asarray_param

        flatten_param_vals = np.asarray([energy.flatten(), mass.flatten(), *[theta_param_mesh.flatten() for theta_param_mesh in theta_params.values()]])
            
        unique_param_vals = np.unique(flatten_param_vals, axis=1)

        logspectralvals = self.point_spectral_gen(energy=unique_param_vals[0], mass=unique_param_vals[1], **{theta_param_key: unique_param_vals[2+idx].flatten() for idx, theta_param_key in enumerate(theta_params.keys())})

        mask = np.all(unique_param_vals[:, None, :] == flatten_param_vals[:, :, None], axis=0)

        slices = np.where(mask, logspectralvals[None, :], 0.0)

        logspectralvals = np.sum(slices, axis=-1).reshape(energy.shape)
            
        return logspectralvals
    

    def mesh_efficient_spectral_gen(self, 
                                    energy: list | np.ndarray | float, 
                                    mass: list | np.ndarray | float, 
                                    theta_params: dict = {'coupling': 0.1}) -> np.ndarray | float:

        mass = np.asarray(mass)

        theta_params = {theta_param_key: np.asarray(theta_param_val) for theta_param_key, theta_param_val in theta_params.items()}

        param_meshes = np.meshgrid(energy, mass, *theta_params.values(), indexing='ij')

        logspectralvals = self.point_spectral_gen(energy = param_meshes[0].flatten(), mass = param_meshes[1].flatten(), **{theta_param_key: param_meshes[2+idx].flatten() for idx, theta_param_key in enumerate(theta_params.keys())}).reshape(param_meshes[0].shape)
        
        return logspectralvals
    
    def __call__(self, *args, **kwargs) -> np.ndarray | float:
        return self.spectral_gen(*args, **kwargs)
        
    
    