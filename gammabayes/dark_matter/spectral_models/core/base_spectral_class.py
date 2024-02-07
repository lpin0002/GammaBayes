import numpy as np
from scipy import interpolate
import pandas as pd
from gammabayes.dark_matter.density_profiles import DM_Profiles
from os import path
ScalarSinglet_Folder_Path = path.dirname(__file__)

from gammabayes.dark_matter.channel_spectra import single_channel_spectral_data_path
from gammabayes.utils import update_with_defaults
import time

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


single_Wchannel_annihilation_ratios   = {}
mass_axis       = np.logspace(-1,2,301)

for channel in darkSUSY_to_PPPC_converter:
    if channel[0]=="W":
        single_Wchannel_annihilation_ratios[channel] = mass_axis*0+1
    else:
        single_Wchannel_annihilation_ratios[channel] = mass_axis*0


# SS_DM_dist(longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile())
class DM_ContinuousEmission_Spectrum(object):
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

    def zero_output(self, inputval):
        """
        A helper method that returns a zero value for any given input. Primarily used as a fallback interpolation function for channels
        that do not have spectral data available.

        Args:
            inputval: Input value or array. The specific value(s) are not used by the function.

        Returns:
            The zero value of the same shape as the input.
        """
        return inputval[0]*0
    def one_output(self, inputval):
        """
        A helper method that returns a one value for any given input. Similar to `zero_output`, but returns one instead of zero, useful
        for certain default or fallback behaviors.

        Args:
            inputval: Input value or array. The specific value(s) are not used by the function.

        Returns:
            A value or array of ones of the same shape as the input.
        """
        return inputval[0]*0 + 1
    
    def __init__(self, 
                 annihilation_fractions=single_Wchannel_annihilation_ratios, 
                 parameter_interpolation_values = [mass_axis], 
                 ratios: bool = True,
                 default_parameter_values = {'mass':1.0,}):
        """
        Initializes the DM_ContinuousEmission_Spectrum class, which calculates the continuous emission spectrum for dark matter annihilation.

        This class leverages interpolated spectral data to provide gamma-ray flux predictions for different dark matter annihilation channels.
        The interpolation is based on pre-calculated tables for a variety of annihilation products, allowing for efficient and accurate
        spectral generation.

        Args:
            annihilation_fractions (dict): A dictionary mapping annihilation channels to their respective fractions. The default uses
                                           ratios for single W-channel annihilation.
            
            parameter_axes (list of np.ndarray): List containing the parameter axes, typically mass axes, over which the annihilation
                                                  fractions are defined. By default, uses a logarithmically spaced mass axis.
            
            ratios (bool): If True, the output values are treated as ratios; otherwise, they are treated as absolute values. Defaults to True.

        The initialization process also involves creating interpolation functions for the square root of the spectral data to ensure
        that the interpolated values remain positive when squared during spectral generation.
        """
        self.ratios = ratios
    
        # This class presumes that you're getting your annihilation ratios from darkSUSY
            # If you're using something else (e.g. MicroOMEGAs) you will need to check that this 
            # dictionary is correct for your case
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

        # Extracting the grid of log10x = log10E - log10mass and mass values (GeV) for the PPPC values
        log10xvals = np.load(single_channel_spectral_data_path+f"/griddata/log10xvals_massenergy_diffflux_grid.npy")
        massvalues = np.load(single_channel_spectral_data_path+f"/griddata/massvals_massenergy_diffflux_grid.npy")


        # We take the square root of the outputs so later we can square them to enforce positivity
        sqrtchannelfuncdictionary = {}
        for darkSUSYchannel in self.darkSUSY_to_PPPC_converter.keys():
            try:
                gammapychannel = self.darkSUSY_to_PPPC_converter[darkSUSYchannel]
                
                # Extracting single channel spectra
                tempspectragrid = np.load(
                    single_channel_spectral_data_path+f"/griddata/channel={gammapychannel}_massenergy_diffflux_grid.npy")
                
                # Interpolating square root of PPPC tables to preserve positivity during interpolation (where result is squared)
                sqrtchannelfuncdictionary[darkSUSYchannel] = interpolate.RegularGridInterpolator(
                    (np.log10(massvalues/1e3), log10xvals), 
                    np.sqrt(np.array(tempspectragrid)), 
                    method='cubic', bounds_error=False, fill_value=0)
            except:
                sqrtchannelfuncdictionary[darkSUSYchannel] = self.zero_output # inputs should be a tuple or list of log_10(mass) in TeV and log_10(x)

        # Saving result to class
        self.sqrtchannelfuncdictionary = sqrtchannelfuncdictionary


        if self.ratios:
            self.normalisations = 0
            for channel in annihilation_fractions.keys():
                self.normalisations += annihilation_fractions[channel]

            self.normalisations = np.where(np.isfinite(self.normalisations), self.normalisations, 0)

            for channel in annihilation_fractions.keys():
                annihilation_fractions[channel] /= self.normalisations


        # Could have done this with the zero and one output functions, 
            # but I'm planning to make this class the head class for others so I don't want to have to re-write this
        if len(parameter_interpolation_values)>1:
            self.partial_sigmav_interpolator_dictionary = {
                channel: interpolate.LinearNDInterpolator(
                    (*parameter_interpolation_values,),
                    annihilation_fractions[channel]) for channel in list(darkSUSY_to_PPPC_converter.keys()
                    )
                    }
        else:
            self.partial_sigmav_interpolator_dictionary = {
                channel: interpolate.interp1d(
                    parameter_interpolation_values[0],
                    annihilation_fractions[channel]) for channel in list(darkSUSY_to_PPPC_converter.keys()
                    )
                    }
            
        self.parameter_interpolation_values = parameter_interpolation_values
        self.parameter_axes = [np.unique(values) for values in self.parameter_interpolation_values]
        self.annihilation_fractions = annihilation_fractions
        self.default_parameter_values = default_parameter_values



    
    def __call__(self, *args, **kwargs) -> np.ndarray | float:
        """
        Allows the instance to be called as a function, which delegates to the `logfunc` method, facilitating easy and intuitive
        usage for calculating the spectrum.

        Args and Returns:
            See `logfunc` method for detailed argument and return value descriptions.
        """
        return self.logfunc(*args, **kwargs)


    def spectral_gen(self, energy: float | np.ndarray | list, 
                           **kwargs) -> np.ndarray | float:
        """
        Generates the dark matter annihilation gamma-ray spectrum for given energy and mass parameters and given dark matter model.

        This method calculates the gamma-ray flux for specified dark matter masses and energy levels by interpolating pre-calculated
        spectral data. The interpolation is performed on the square root of the spectral data to maintain the non-negativity of the flux.

        Args:
            energy (float | np.ndarray | list): Gamma-ray energies for which to calculate the spectrum, in TeV.
            
            mass (float | np.ndarray | list, optional): Dark matter mass in TeV. Defaults to 1.0 TeV.
            
            **kwargs: Additional keyword arguments that can be used to specify other model parameters.

        Returns:
            np.ndarray | float: The gamma-ray flux for the specified energies and dark matter mass, potentially including additional
                                model parameters.
        """
        
        logspectra = -np.inf

        update_with_defaults(kwargs, self.default_parameter_values)

        for channel in self.sqrtchannelfuncdictionary.keys():

            channel_sigma = self.partial_sigmav_interpolator_dictionary[channel](*kwargs.values())
            channel_spectrum = (self.sqrtchannelfuncdictionary[channel]((np.log10(kwargs['mass']), 
                                                           np.log10(energy)-np.log10(kwargs['mass']))))**2 # Square is to enforce positivity

                            
            channel_comp = channel_sigma*channel_spectrum 

            logspectra = np.logaddexp(
                logspectra, 
                np.log(channel_comp))
        
        return logspectra

    
        
    def logfunc(self, 
                energy: list | np.ndarray | float, 
                kwd_parameters: dict = {'mass':1.0, 'coupling': 0.1}) -> np.ndarray | float:
        """
        Calculates the logarithm of the gamma-ray flux for specified energy values and keyword parameters defining the dark matter model.

        This method serves as a wrapper around `spectral_gen`, facilitating easy calculation of the gamma-ray flux's logarithm. It is
        designed to work with a flexible number of keyword arguments that define the dark matter model parameters.

        Args:
            energy (list | np.ndarray | float): Gamma-ray energies for which to calculate the log flux, in TeV.
            kwd_parameters (dict, optional): A dictionary of keyword parameters for the dark matter model, including mass and coupling values.
                                             Defaults to {'mass':1.0, 'coupling': 0.1}.

        Returns:
            np.ndarray | float: The log of the gamma-ray flux for the specified energies and dark matter model parameters.
        """
        energy = np.asarray(energy)

        # for key, val in kwd_parameters.items():
        #     asarray_param = np.asarray(val) 
        #     kwd_parameters[key] = asarray_param

        # flatten_param_vals = np.asarray([energy.flatten(), *[theta_param_mesh.flatten() for theta_param_mesh in kwd_parameters.values()]])
            
        # unique_param_vals = np.unique(flatten_param_vals, axis=1)

        # logspectralvals = self.spectral_gen(
        #     energy=unique_param_vals[0], 
        #     **{param_key: unique_param_vals[1+idx].flatten() for idx, param_key in enumerate(kwd_parameters.keys())})

        # mask = np.all(unique_param_vals[:, None, :] == flatten_param_vals[:, :, None], axis=0)

        # slices = np.where(mask, logspectralvals[None, :], 0.0)

        # logspectralvals = np.sum(slices, axis=-1).reshape(energy.shape)

        logspectralvals = self.spectral_gen(
            energy=energy, 
            **kwd_parameters)            
        return logspectralvals
    

    # This function presumes that it needs to create a mesh based on the input parameters
        # this is handy when one doesn't want to create a mesh that includes all the observation
        # parameter axes, spectral parameters, and spatial parameters at once, reducing dimensionality
        # and reduces the number of needed computations
    def mesh_efficient_logfunc(self, 
                               energy: list | np.ndarray | float, 
                               kwd_parameters: dict = {'mass':1.0, 'coupling': 0.1}) -> np.ndarray | float:
        """
        An mesh efficient version of `logfunc` that utilizes meshgrid computations for generating the dark matter annihilation gamma-ray spectrum.

        This method is optimized for scenarios where computations over a grid of model parameters are required, reducing the computational
        overhead by leveraging numpy's meshgrid functionality. It calculates the gamma-ray flux for each combination of energy and model
        parameters specified in `kwd_parameters`.

        Args:
            energy (list | np.ndarray | float): Gamma-ray energies for which to calculate the log flux, in TeV.
            kwd_parameters (dict, optional): A dictionary of keyword parameters for the dark matter model, including mass and coupling values.
                                             Defaults to {'mass':1.0, 'coupling': 0.1}.

        Returns:
            np.ndarray | float: The log of the gamma-ray flux for the specified energies and dark matter model parameters over the grid
                                defined by the input parameters.
        """


        kwd_parameters = {param_key: np.asarray(param_val) for param_key, param_val in kwd_parameters.items()}

        param_meshes = np.meshgrid(energy, *kwd_parameters.values(), indexing='ij')

        logspectralvals = self.spectral_gen(
            energy = param_meshes[0].flatten(), 
            **{param_key: param_meshes[1+idx].flatten() for idx, param_key in enumerate(kwd_parameters.keys())}
            ).reshape(param_meshes[0].shape)
        
        return logspectralvals




