import numpy as np
from scipy import interpolate
import pandas as pd
from gammabayes.dark_matter.density_profiles import DM_Profiles
from os import path
ScalarSinglet_Folder_Path = path.dirname(__file__)

from gammabayes.dark_matter.channel_spectra import (
    single_channel_spectral_data_path,
    PPPCReader, 
)

from gammabayes.utils import update_with_defaults
import time

from decimal import Decimal, getcontext

# Set the precision higher than default
getcontext().prec = 600  # Setting precision to 600 decimal places



single_Wchannel_annihilation_ratios   = {}
mass_axis       = np.logspace(-1,2,301)

for channel in PPPCReader.darkSUSY_to_PPPC_converter:
    if channel[0]=="W":
        single_Wchannel_annihilation_ratios[channel] = mass_axis*0+1
    else:
        single_Wchannel_annihilation_ratios[channel] = mass_axis*0


# SS_DM_dist(longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile())
class DM_ContinuousEmission_Spectrum(object):


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
                 parameter_interpolation_values:  list[np.ndarray] = [mass_axis], 
                 ratios: bool = True,
                 default_parameter_values = {'mass':1.0,},
                 ):
        """
        Initialize the DM_ContinuousEmission_Spectrum class to compute the continuous emission spectrum from dark matter annihilation across various channels. This class utilizes precomputed spectral data for different annihilation channels and interpolates these data to provide gamma-ray flux predictions for any given dark matter mass and energy.

        Args:
            annihilation_fractions (dict): A dictionary mapping dark matter annihilation channels to their respective fraction of total annihilation events. This parameter defines the contribution of each channel to the overall annihilation spectrum.
            
            parameter_interpolation_values (dict | list): A structure specifying the parameter values over which interpolation should be performed. If a dictionary, keys should match the expected parameter names (e.g., 'mass'), with values being arrays of parameter values. If a list, it is expected to contain arrays of parameter values directly, assuming the order matches `parameter_names` if provided.
            
            ratios (bool, optional): Indicates whether the provided annihilation fractions are to be interpreted as ratios (True) or absolute values (False). When True, fractions will be normalized to sum to 1 across all channels. Defaults to True.
            
            default_parameter_values (dict, optional): A dictionary specifying default values for any parameters required for generating the spectrum but not included in `parameter_interpolation_values`. This could include parameters like 'mass' if not otherwise specified.
            
            parameter_names (list, optional): An optional list of parameter names corresponding to the arrays provided in `parameter_interpolation_values` when it is a list. This is used to ensure proper mapping of values to parameters during interpolation. If `parameter_interpolation_values` is a dictionary, this argument is ignored.

        This class supports the interpolation of spectral data across a range of dark matter masses and other parameters, allowing for flexible and dynamic spectral analysis. It handles both single and multidimensional interpolation based on the provided `parameter_interpolation_values`, adjusting automatically to the complexity of the input parameter space.
        """
        self.ratios = ratios
    
        # This class presumes that you're getting your annihilation ratios from darkSUSY
            # If you're using something else (e.g. MicroOMEGAs) you will need to check that this 
            # dictionary is correct for your case


        atprod_gammas = PPPCReader(single_channel_spectral_data_path+"/PPPC_Tables/AtProduction_gamma_EW_corrections.dat")
        atprod_mass_values = atprod_gammas.mass_axis
        atprod_log10x_values = atprod_gammas.log10x_axis

        # We take the square root of the outputs so later we can square them to enforce positivity
        sqrtchannelfuncdictionary = {}
        for darkSUSYchannel in PPPCReader.darkSUSY_to_PPPC_converter.keys():
            try:
                PPPC_channel = PPPCReader.darkSUSY_to_PPPC_converter[darkSUSYchannel]
                
                # Extracting single channel spectra
                tempspectragrid = atprod_gammas[PPPC_channel].reshape(atprod_gammas.output_shape)
                
                # Interpolating square root of PPPC tables to preserve positivity during interpolation (where result is squared)
                sqrtchannelfuncdictionary[darkSUSYchannel] = interpolate.RegularGridInterpolator(
                    (np.log10(atprod_mass_values), atprod_log10x_values), 
                    np.sqrt(np.asarray(tempspectragrid)/1000), # 1000 is to convert to 1/TeV 
                    method='cubic', bounds_error=False, fill_value=0)
            except:
                sqrtchannelfuncdictionary[darkSUSYchannel] = self.zero_output # inputs should be a tuple or list of log_10(mass) in TeV and log_10(x)

        # Saving result to class
        self.sqrtchannelfuncdictionary = sqrtchannelfuncdictionary


        if self.ratios:
            self.log_normalisations = -np.inf
            for channel in annihilation_fractions.keys():
                self.log_normalisations = np.logaddexp(self.log_normalisations, np.log(annihilation_fractions[channel]))

            self.log_normalisations = np.where(np.isfinite(self.log_normalisations), self.log_normalisations, 0)

            for channel in annihilation_fractions.keys():
                new_annihilation_fractions = np.exp(np.log(annihilation_fractions[channel])-self.log_normalisations)
                annihilation_fractions[channel] = new_annihilation_fractions


        
        
        # sqrt enforces positivity while also transforming values to closer to 1
        if len(parameter_interpolation_values)>1:
            self.partial_sqrt_sigmav_interpolator_dictionary = {
                channel: interpolate.RegularGridInterpolator(
                    (*parameter_interpolation_values,),
                    np.sqrt(annihilation_fractions[channel]),
                    # 'cubic' method can be unstable for small values, we highly recommend 
                        # you use ratios=True unless otherwise required
                    method='cubic', bounds_error=False, fill_value=0) \
                        for channel in list(PPPCReader.darkSUSY_to_PPPC_converter.keys()
                    )
                    }
        else:
            self.partial_sqrt_sigmav_interpolator_dictionary = {
                channel: interpolate.interp1d(
                    parameter_interpolation_values[0],
                    np.sqrt(annihilation_fractions[channel]),
                    kind='cubic', bounds_error=False, fill_value='extrapolate') for channel in list(PPPCReader.darkSUSY_to_PPPC_converter.keys()
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

            channel_sigma = self.partial_sqrt_sigmav_interpolator_dictionary[channel]((*kwargs.values(),))**2
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




