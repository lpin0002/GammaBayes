import numpy as np
from scipy import interpolate
import pandas as pd
from gammabayes.dark_matter.density_profiles import DM_Profiles
from os import path
ScalarSinglet_Folder_Path = path.dirname(__file__)

from gammabayes.dark_matter.channel_spectra import single_channel_spectral_data_path
import time

# SS_DM_dist(longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile())
class Z2_ScalarSinglet(object):
    
    def __init__(self, ratios: bool = True):

        self.ratios = ratios
    
        # This class presumes that you're getting your annihilation ratios from darkSUSY
            # If you're using something else (e.g. MicroOMEGAs) you will need to check that this 
            # dictionary is correct for your case
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

        # Extracting the annihilation ratios for the Scalar Singlet model
        darkSUSY_BFs_cleaned = pd.read_csv(ScalarSinglet_Folder_Path+'/darkSUSY_BFs/darkSUSY_BFs_cleaned.csv', delimiter=' ')

        # darkSUSY works in GeV, so this line converts to TeV
        darkSUSY_massvalues = darkSUSY_BFs_cleaned.iloc[:,1]/1e3

        # Extracting the Higgs coupling values
        darkSUSY_lambdavalues = darkSUSY_BFs_cleaned.iloc[:,2]



       
        # Extracting the grid of log10x = log10E - log10mass and mass values (GeV) for the PPPC values
        log10xvals = np.load(single_channel_spectral_data_path+f"/griddata/log10xvals_massenergy_diffflux_grid.npy")
        massvalues = np.load(single_channel_spectral_data_path+f"/griddata/massvals_massenergy_diffflux_grid.npy")


        # We take the square root of the outputs so later we can square them to enforce positivity
        sqrtchannelfuncdictionary = {}
        for darkSUSYchannel in darkSUSY_to_PPPC_converter.keys():
            try:
                gammapychannel = darkSUSY_to_PPPC_converter[darkSUSYchannel]
                
                # Extracting single channel spectra
                tempspectragrid = np.load(
                    single_channel_spectral_data_path+f"/griddata/channel={gammapychannel}_massenergy_diffflux_grid.npy")
                
                # Interpolating square root of PPPC tables to preserve positivity during interpolation (where result is squared)
                sqrtchannelfuncdictionary[darkSUSYchannel] = interpolate.RegularGridInterpolator(
                    (np.log10(massvalues/1e3), log10xvals), 
                    np.sqrt(np.array(tempspectragrid)), 
                    method='cubic', bounds_error=False, fill_value=0)
            except:
                sqrtchannelfuncdictionary[darkSUSYchannel] = lambda inputs: inputs[0]*0 # inputs should be a tuple or list of log_10(mass) in TeV and log_10(x)


        # Saving result to class
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
    
    def __call__(self, *args, **kwargs) -> np.ndarray | float:
        return self.logfunc(*args, **kwargs)


    def spectral_gen(self, energy: float | np.ndarray | list, 
                           mass: float | np.ndarray | list = 1.0, 
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
        
        logspectra = -np.inf
        for channel in self.sqrtchannelfuncdictionary.keys():
            logspectra = np.logaddexp(
                logspectra, 
                np.log(
                    self.partial_sigmav_interpolator_dictionary[channel](mass, coupling)\
                        *(self.sqrtchannelfuncdictionary[channel]((np.log10(mass), 
                        np.log10(energy)-np.log10(mass))))**2)) # Square is to enforce positivity
        
        return logspectra

    
        
    def logfunc(self, 
                energy: list | np.ndarray | float, 
                kwd_parameters: dict = {'mass':1.0, 'coupling': 0.1}) -> np.ndarray | float:
        energy = np.asarray(energy)

        for key, val in kwd_parameters.items():
            asarray_param = np.asarray(val) 
            kwd_parameters[key] = asarray_param

        flatten_param_vals = np.asarray([energy.flatten(), *[theta_param_mesh.flatten() for theta_param_mesh in kwd_parameters.values()]])
            
        unique_param_vals = np.unique(flatten_param_vals, axis=1)

        logspectralvals = self.spectral_gen(
            energy=unique_param_vals[0], 
            **{param_key: unique_param_vals[1+idx].flatten() for idx, param_key in enumerate(kwd_parameters.keys())})

        mask = np.all(unique_param_vals[:, None, :] == flatten_param_vals[:, :, None], axis=0)

        slices = np.where(mask, logspectralvals[None, :], 0.0)

        logspectralvals = np.sum(slices, axis=-1).reshape(energy.shape)
            
        return logspectralvals
    

    def mesh_efficient_logfunc(self, 
                               energy: list | np.ndarray | float, 
                               kwd_parameters: dict = {'mass':1.0, 'coupling': 0.1}) -> np.ndarray | float:


        kwd_parameters = {param_key: np.asarray(param_val) for param_key, param_val in kwd_parameters.items()}

        param_meshes = np.meshgrid(energy, *kwd_parameters.values(), indexing='ij')

        logspectralvals = self.spectral_gen(
            energy = param_meshes[0].flatten(), 
            **{param_key: param_meshes[1+idx].flatten() for idx, param_key in enumerate(kwd_parameters.keys())}
            ).reshape(param_meshes[0].shape)
        
        return logspectralvals




