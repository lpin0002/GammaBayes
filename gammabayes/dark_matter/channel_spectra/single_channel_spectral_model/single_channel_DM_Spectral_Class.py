import numpy as np
from scipy import interpolate
import os

single_channel_spectral_data_path = os.path.dirname(os.path.dirname(__file__))

from ..PPPC_Tables import PPPCReader

from gammabayes import update_with_defaults


class SingleDMChannel(object):
    """Function that allows for efficient single channel dark matter spectra calculations."""

    
    def __init__(self, channel='W+W-',
                 default_parameter_values = {'mass':1.0,},
                 ):
    
        self.channel = channel
        atprod_gammas = PPPCReader(single_channel_spectral_data_path+"/PPPC_Tables/AtProduction_gamma_EW_corrections.dat")
        atprod_mass_values = atprod_gammas.mass_axis
        atprod_log10x_values = atprod_gammas.log10x_axis

        # We take the square root of the outputs so later we can square them to enforce positivity
        try:
            PPPC_channel = PPPCReader.darkSUSY_to_PPPC_converter[channel]
        
            # Extracting single channel spectra
            tempspectragrid = atprod_gammas[PPPC_channel].reshape(atprod_gammas.output_shape)
            
        except:
            # Extracting single channel spectra
            tempspectragrid = atprod_gammas[channel].reshape(atprod_gammas.output_shape)
            
        # Interpolating square root of PPPC tables to preserve positivity during interpolation (where result is squared)
        self.sqrtchannelfunc = interpolate.RegularGridInterpolator(
            (np.log10(atprod_mass_values), atprod_log10x_values), 
            np.sqrt(np.asarray(tempspectragrid)/1000), # 1000 is to convert to 1/TeV 
            method='cubic', bounds_error=False, fill_value=0)


        self.default_parameter_values = default_parameter_values


    
    def __call__(self, *args, **kwargs) -> np.ndarray | float:
        return self.logfunc(*args, **kwargs)


    def spectral_gen(self, energy: float | np.ndarray | list, 
                           **kwargs) -> np.ndarray | float:
        
        update_with_defaults(kwargs, self.default_parameter_values)


        log_channel_spectrum = (self.sqrtchannelfunc((np.log10(kwargs['mass']), 
                                                        np.log10(energy)-np.log10(kwargs['mass']))))**2 # Square is to enforce positivity
            
        
        return np.log(log_channel_spectrum)

    
        
    def logfunc(self, 
                energy: list | np.ndarray | float, 
                kwd_parameters: dict = {'mass':1.0}) -> np.ndarray | float:

        energy = np.asarray(energy)

        for key, val in kwd_parameters.items():
            kwd_parameters[key] = np.asarray(val) 

        flatten_param_vals = np.asarray([energy.flatten(), *[theta_param.flatten() for theta_param in kwd_parameters.values()]])
            
        unique_param_vals = np.unique(flatten_param_vals, axis=1)

        logspectralvals = self.spectral_gen(
            energy=unique_param_vals[0], 
            **{param_key: unique_param_vals[1+idx].flatten() for idx, param_key in enumerate(kwd_parameters.keys())})

        mask = np.all(unique_param_vals[:, None, :] == flatten_param_vals[:, :, None], axis=0)

        slices = np.where(mask, logspectralvals[None, :], 0.0)

        logspectralvals = np.sum(slices, axis=-1).reshape(energy.shape)
        
        return logspectralvals
    

    # This function presumes that it needs to create a mesh based on the input parameters
        # this is handy when one doesn't want to create a mesh that includes all the observation
        # parameter axes, spectral parameters, and spatial parameters at once, reducing dimensionality
        # and reduces the number of needed computations
    def mesh_efficient_logfunc(self, 
                               energy: list | np.ndarray | float, 
                               kwd_parameters: dict = {'mass':1.0}) -> np.ndarray | float:


        new_kwd_parameters = {param_key: np.asarray(param_val) for param_key, param_val in kwd_parameters.items()}


        param_meshes = np.meshgrid(energy, *new_kwd_parameters.values(), indexing='ij')

        logspectralvals = self.spectral_gen(
            energy = param_meshes[0].flatten(), 
            **{param_key: param_meshes[1+idx].flatten() for idx, param_key in enumerate(new_kwd_parameters.keys())}
            ).reshape(param_meshes[0].shape)
        
        return logspectralvals



    def mesh_integral_efficient_logfunc(self, 
                               energy: list | np.ndarray | float, 
                               kwd_parameters: dict = {'mass':1.0}) -> np.ndarray | float:

        energy = np.asarray(energy)

        new_kwd_parameters = {param_key: np.asarray(param_val) for param_key, param_val in kwd_parameters.items()}


        for key, val in new_kwd_parameters.items():
            new_kwd_parameters[key] = np.asarray(val) 

        hyper_params_shape = new_kwd_parameters[list(new_kwd_parameters.keys())[0]].shape
        # minimises the amount of needed memory and comp time by reducing number of combinations in meshgrid
        flatten_hyperparam_vals = np.asarray([*[theta_param.flatten() for theta_param in new_kwd_parameters.values()]])
            
        unique_param_vals = np.unique(flatten_hyperparam_vals, axis=1)

        param_meshes = np.meshgrid(energy, *unique_param_vals, indexing='ij')


        logspectralvals = self.spectral_gen(
            energy = param_meshes[0].flatten(), 
            **{param_key: param_meshes[1+idx].flatten() for idx, param_key in enumerate(new_kwd_parameters.keys())}
            ).reshape(param_meshes[0].shape)
        
        mask = np.all(unique_param_vals[:, None, :] == flatten_hyperparam_vals[:, :, None], axis=0)

        slices = np.where(mask, logspectralvals[:, None, :], 0.0)


        logspectralvals = np.sum(slices, axis=-1).reshape((energy.size, *hyper_params_shape))


        return logspectralvals
    

    def calc_ratios(self, *args, **kwargs):

        return {self.channel, 1.0}







