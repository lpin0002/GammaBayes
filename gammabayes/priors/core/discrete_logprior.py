from scipy.special import logsumexp
import numpy as np, time, warnings, logging, pickle
from gammabayes.samplers import  integral_inverse_transform_sampler
from gammabayes.utils import iterate_logspace_integration, construct_log_dx_mesh
from gammabayes import update_with_defaults, GammaObs, GammaBinning, GammaLogExposure
# from gammabayes import EventData
from icecream import ic
from astropy import units as u
import matplotlib.pyplot as plt

class DiscreteLogPrior(object):
    """
    A class representing a discrete log prior distribution.

    Args:
        name (str, optional): Name of the instance. Defaults to '[None]'.
        inputunits (str, optional): Unit of the input values for the axes. Defaults to None.
        logfunction (callable, optional): A function that calculates log prior values given axes values and hyperparameters.
        log_mesh_efficient_func (callable, optional): A function for efficient computation of log prior values on a mesh grid.
        axes (tuple[np.ndarray], optional): A tuple containing np.ndarray objects for each axis over which the prior is defined.
        axes_names (list[str] | tuple[str], optional): Names of the axes. Defaults to ['None'].
        default_spectral_parameters (dict, optional): Default spectral parameters for the prior. Defaults to an empty dict.
        default_spatial_parameters (dict, optional): Default spatial parameters for the prior. Defaults to an empty dict.
        iterative_logspace_integrator (callable, optional): Function used for integrations in log space. Defaults to iterate_logspace_integration.
    """
    
    def __init__(self, 
                 name: str=None, 
                 inputunits: str=None, 
                 logfunction: callable=None, 
                 log_mesh_efficient_func: callable = None,
                 axes: tuple[np.ndarray] | None = None, 
                 binning_geometry: GammaBinning = None,
                 default_spectral_parameters: dict = {},  
                 default_spatial_parameters: dict = {},  
                 iterative_logspace_integrator: callable = iterate_logspace_integration,
                 irf_loglike=None,
                 log_scaling_factor: int|float =0.,
                 ):
        """
        Initializes a DiscreteLogPrior object, which represents a discrete log prior distribution.

        Args:
            name (str, optional): Name of the instance. Defaults to None.
            inputunits (str, optional): Unit of the input values for the axes. Defaults to None.
            logfunction (callable, optional): A function that calculates log prior values given axes values and hyperparameters.
            log_mesh_efficient_func (callable, optional): A function for efficient computation of log prior values on a mesh grid.
            axes (tuple[np.ndarray], optional): A tuple containing np.ndarray objects for each axis over which the prior is defined.
            axes_names (list[str] | tuple[str], optional): Names of the axes. Defaults to ['None'].
            default_spectral_parameters (dict, optional): Default spectral parameters for the prior. Defaults to an empty dict.
            default_spatial_parameters (dict, optional): Default spatial parameters for the prior. Defaults to an empty dict.
            iterative_logspace_integrator (callable, optional): Function used for integrations in log space. Defaults to iterate_logspace_integration.

            
        Note:
        - This class assumes the prior is defined in a discrete log space along specified axes.
        - The axes should correspond to physical quantities over which the prior is distributed, such as energy and sky coordinates.
        """

        if name is None:
            name = time.strftime("discrete_log_prior_%f")
        self.name = name
        self.log_scaling_factor = log_scaling_factor

        self.inputunits = inputunits
        self._logfunction = logfunction
        self.irf_loglike = irf_loglike

        self._create_geometry(axes=axes, binning_geometry=binning_geometry)

            

        if not(log_mesh_efficient_func is None):
            self.efficient_exist = True
            self._log_mesh_efficient_func = log_mesh_efficient_func
            self.log_mesh_efficient_func = self._scaled_log_mesh_efficient_func
        else:
            self.efficient_exist = False

    
            
        self.default_spectral_parameters = default_spectral_parameters
        self.default_spatial_parameters = default_spatial_parameters

        self.num_spec_params = len(default_spectral_parameters)
        self.num_spat_params = len(default_spatial_parameters)


        self.logspace_integrator = iterative_logspace_integrator
            

    def _create_geometry(self, axes, binning_geometry):


        if not(axes is None):
            self.binning_geometry = GammaBinning(energy_axis=axes[0], lon_axis=axes[1], lat_axis=axes[2])
        elif not(binning_geometry is None):
            self.binning_geometry = binning_geometry
        else:
            self.binning_geometry = None

        
    def logfunction(self, *args, **kwargs):
        return  self.log_scaling_factor + self._logfunction(*args, **kwargs)
    
    
    def _scaled_log_mesh_efficient_func(self, *args, **kwargs):

        return  self.log_scaling_factor + self._log_mesh_efficient_func(*args, **kwargs)

    
    
    def __repr__(self) -> str:
        """
        String representation of the DiscreteLogPrior instance.

        Returns:
            str: A description of the instance including its name, logfunction type, input units, and axes names.
        """
        description = f"Discrete log prior class\n{'-' * 20}\n" \
                      f"Name: {self.name}\n" \
                      f"Logfunction type: {type(self.logfunction).__name__}\n" \
                      f"Input units: {self.inputunits}\n" 
        return description
    
    
    def __call__(self, *args, **kwargs)  -> np.ndarray | float:
        """
        Allows the instance to be called like a function, passing arguments directly to the logfunction.

        Args:
            *args: Arguments for the logfunction.
            **kwargs: Keyword arguments for the logfunction.

        Returns:
            np.ndarray | float: The result from the logfunction, which is the log prior value(s) for the given input(s).
        """
        output = self.logfunction(*args, **kwargs)
        return output

    
    def log_normalisation(self, log_prior_values: np.ndarray = None, 
                      spectral_parameters: dict = {}, 
                      spatial_parameters: dict = {},
                      axisindices: list = [0,1,2],
                      *args, **kwargs) -> np.ndarray | float:
        """
        Calculates the log normalisation constant of the log prior over specified axes.

        Args:
            log_prior_values (np.ndarray, optional): Pre-computed log prior values. If None, they will be computed using default or provided hyperparameters.
            spectral_parameters (dict, optional): Spectral parameters to be used if log_prior_values is not provided. Defaults to instance's default spectral parameters.
            spatial_parameters (dict, optional): Spatial parameters to be used if log_prior_values is not provided. Defaults to instance's default spatial parameters.
            axisindices (list, optional): Indices of the axes over which to integrate. Defaults to [0, 1, 2].

        Returns:
            np.ndarray | float: The normalisation constant for the log prior, either as a scalar or an array depending on the integration over multiple axes.
        """
        

        if (log_prior_values is []) | (log_prior_values is None):
            # Checks if spectral_parameters is an empty dict. If so, it sets it to the defaults
            update_with_defaults(spectral_parameters, self.default_spectral_parameters)
            update_with_defaults(spatial_parameters, self.default_spatial_parameters)

            if self.efficient_exist:
                log_prior_values = self.log_mesh_efficient_func(*self.binning_geometry.axes, 
                                                                spectral_parameters = spectral_parameters,
                                                                spatial_parameters = spatial_parameters,
                                                                *args, **kwargs)
            else:
                inputmesh = np.meshgrid(*self.binning_geometry.axes, 
                                *spectral_parameters.values(),
                                *spatial_parameters.values(), indexing='ij') 
        
                log_prior_values = self.logfunction(*inputmesh[:self.binning_geometry.num_axes], 
                                                spectral_parameters = {hyper_key: inputmesh[self.binning_geometry.num_axes+idx] for idx, hyper_key in enumerate(spectral_parameters.keys())}, 
                                                spatial_parameters = {hyper_key: inputmesh[self.binning_geometry.num_axes+len(spectral_parameters)+idx] for idx, hyper_key in enumerate(spatial_parameters.keys())},
                                                *args, **kwargs
                                                )    
        log_prior_norms = self.logspace_integrator(logy=np.squeeze(log_prior_values), axes=[axis.value for axis in self.binning_geometry.axes], axisindices=axisindices)

        return log_prior_norms
    
    
    
    def sample(self, 
               numsamples: int=None, 
               log_prior_values: np.ndarray = None, 
               spectral_parameters: dict = None, 
               spatial_parameters: dict = None,
               *args, **kwargs,
               )  -> np.ndarray:
        """
        Generates samples from the prior distribution using inverse transform sampling.

        Parameters:
        - numsamples (int): Number of samples to generate.
        
        - log_prior_values (np.ndarray, optional): Log prior values to sample from. If None, they are computed using the provided or default parameters.
        
        - spectral_parameters (dict, optional): Spectral parameters for computing log prior values. Defaults to instance's parameters.
        
        - spatial_parameters (dict, optional): Spatial parameters for computing log prior values. Defaults to instance's parameters.

        Returns:
        - np.ndarray: Samples from the prior, with shape determined by the number of axes and samples requested.
        """
        if spectral_parameters is None:
            spectral_parameters = self.default_spectral_parameters
        if spatial_parameters is None:
            spatial_parameters = self.default_spatial_parameters

        update_with_defaults(spectral_parameters, self.default_spectral_parameters)
        update_with_defaults(spatial_parameters, self.default_spatial_parameters)
            
        
            
        
        if log_prior_values is None:
            if self.efficient_exist:
                log_prior_values = self.log_mesh_efficient_func(*self.binning_geometry.axes, 
                                                                spectral_parameters = spectral_parameters,
                                                                spatial_parameters = spatial_parameters,
                                                                *args, **kwargs)

            else:
                inputmesh = np.meshgrid(*self.binning_geometry.axes, 
                                *spectral_parameters.values(),
                                *spatial_parameters.values(), indexing='ij') 
        
                log_prior_values = self.logfunction(*inputmesh[:self.binning_geometry.num_axes], 
                                                spectral_parameters = {hyper_key: inputmesh[self.binning_geometry.num_axes+idx] for idx, hyper_key in enumerate(spectral_parameters.keys())}, 
                                                spatial_parameters = {hyper_key: inputmesh[self.binning_geometry.num_axes+len(spectral_parameters)+idx] for idx, hyper_key in enumerate(spatial_parameters.keys())},
                                                *args, **kwargs
                                                )
    
        log_prior_values = np.asarray(log_prior_values)
                
                
                
        # This code is presuming a large number of events. This can cause a lot of numerical instability issues down the line 
            # of a hierarchical models (especially without the use of samplers which is currently the case for this code)
            # So we will double check the normalisation
        log_normalisation = self.log_normalisation(log_prior_values=log_prior_values, *args, **kwargs)


        if numsamples is None:
            # If no samples are given it is presumed that the prior function corresponds to the observed flux rate
                # In which case the normalisation corresponds to the number of events for the given exposure
            numsamples = np.exp(log_normalisation)


        numsamples = int(round(numsamples))

        
        if numsamples>=1:

            log_prior_values = np.squeeze(log_prior_values) - log_normalisation

            logdx = construct_log_dx_mesh(self.binning_geometry.axes)
                        
            simvals = integral_inverse_transform_sampler(log_prior_values+logdx, axes=self.binning_geometry.axes, Nsamples=numsamples)
            
            
                

        else:
            simvals = [[], [], []]
        

        if hasattr(self, 'log_exposure_map'):
            log_exposure = self.log_exposure_map
        else:
            log_exposure = None
        
        if hasattr(self, 'pointing_dir'):
            pointing_dir = self.pointing_dir
        else:
            pointing_dir = None

        if hasattr(self, 'observation_time'):
            observation_time = self.observation_time
        else:
            observation_time = None



        return GammaObs(energy=simvals[0], 
                        lon=simvals[1], 
                        lat=simvals[2], 
                        binning_geometry=self.binning_geometry,
                        meta={'source':self.name},
                        irf_loglike=self.irf_loglike,
                        pointing_dir=pointing_dir,
                        observation_time=observation_time,
                        log_exposure=log_exposure
                        )
    
    def construct_prior_array(self, 
                              spectral_parameters: dict = {}, 
                              spatial_parameters: dict = {}, 
                              normalisation_axes: list | tuple = [0,1,2],
                              normalise: bool = False,)  -> np.ndarray:
        """
        Constructs an array of log prior values over a mesh of the axes' values.

        e.g. for the mesh axis1=[0,1] and axis2=[0,2] then mesh = [ [[0,0], [0,2]], [[1,0], [1,2]] ]

        Parameters:
        - spectral_parameters (dict, optional): Spectral parameters to use. Defaults to instance's default parameters.
        
        - spatial_parameters (dict, optional): Spatial parameters to use. Defaults to instance's default parameters.
        
        - normalisation_axes (list | tuple, optional): Axes indices over which to normalise. Defaults to [0, 1, 2].
        
        - normalise (bool, optional): Whether to normalise the output array. Defaults to False.

        Returns:
        - np.ndarray: An array of log prior values for the specified parameters and axes.
        """
        update_with_defaults(spectral_parameters, self.default_spectral_parameters)
        update_with_defaults(spatial_parameters, self.default_spatial_parameters)

        if self.efficient_exist:

            outputarray = self.log_mesh_efficient_func(*self.binning_geometry.axes, 
                                                       spectral_parameters=spectral_parameters, 
                                                       spatial_parameters=spatial_parameters)
            

        else:
            inputmesh = np.meshgrid(*self.binning_geometry.axes, 
                                    *spectral_parameters.values(),
                                    *spatial_parameters.values(), indexing='ij') 
            
            outputarray = self.log_scaling_factor+self.logfunction(*inputmesh[:self.binning_geometry.num_axes], 
                                            spectral_parameters = {hyper_key: inputmesh[self.binning_geometry.num_axes+idx] for idx, hyper_key in enumerate(spectral_parameters.keys())}, 
                                            spatial_parameters = {hyper_key: inputmesh[self.binning_geometry.num_axes+len(spectral_parameters)+idx] for idx, hyper_key in enumerate(spatial_parameters.keys())}
                                            )
    
        if normalise:
            # Normalisation is done twice to reduce numerical instability issues
            log_normalisation = self.log_normalisation(log_prior_values = outputarray, axisindices=normalisation_axes)

            logging.info(f"normalisation.shape: {log_normalisation.shape}")
            log_normalisation = np.where(np.isinf(log_normalisation), 0, log_normalisation)
            outputarray = outputarray - log_normalisation

            # normalisation = self.normalisation(log_prior_values = outputarray, axisindices=normalisation_axes)
            # normalisation = np.where(np.isneginf(normalisation), 0, normalisation)
            # outputarray = outputarray - normalisation

             
        return outputarray
    

    def save(self, file_name:str, write_method='wb' ):
        """
        Saves the DiscreteLogPrior data to a pkl file.

        Args:
        file_name (str): The name of the file to save the data to.
        """

        if not(file_name.endswith('.pkl')):
            file_name = file_name+'.pkl'

        pickle.dump(self, open(file_name,write_method))

    @classmethod
    def load(cls, file_name:str):
        """_summary_

        Args:
            file_name (str): _description_

        Returns:
            _type_: _description_
        """
        if not(file_name.endswith(".pkl")):
            file_name = file_name + ".pkl"
        return  pickle.load(open(file_name,'rb'))



    def rescale(self, log_factor:float|int=0.):
        self.log_scaling_factor = self.log_scaling_factor+log_factor


    def peek(self, vmin=None, vmax=None, norm='linear', cmap='viridis', **kwargs):
        from matplotlib import pyplot as plt
        from gammabayes.utils.integration import iterate_logspace_integration
        from matplotlib.pyplot import get_cmap

        log_matrix_values = self.construct_prior_array()

        cmap = get_cmap(cmap)



        kwargs.setdefault('figsize', (12,6))
        fig, ax = plt.subplots(1,3, **kwargs)

        log_integrated_energy_flux = iterate_logspace_integration(logy=log_matrix_values, 
                                                axes=[self.binning_geometry.lon_axis.value, 
                                                      self.binning_geometry.lat_axis.value],
                                                axisindices=[1,2])

        log_integrated_spatial_flux = iterate_logspace_integration(logy=log_matrix_values, 
                                                axes=[self.binning_geometry.energy_axis.value],
                                                axisindices=[0])
        

        log_integrated_mix_flux = iterate_logspace_integration(logy=log_matrix_values, 
                                                axes=[self.binning_geometry.lat_axis.value],
                                                axisindices=[2])


        ax[0].plot(self.binning_geometry.energy_axis.value, np.exp(log_integrated_energy_flux), c=cmap(0.5))
        ax[0].set_xscale('log')
        ax[0].set_yscale(norm)
        
        ax[0].set_xlabel(f'Energy [{self.binning_geometry.energy_axis.unit.to_string()}]')
        ax[0].grid(which='major', c='grey', ls='--', alpha=0.4)


        pcm = ax[1].pcolormesh(self.binning_geometry.lon_axis.value, 
                        self.binning_geometry.lat_axis.value, 
                        np.exp(log_integrated_spatial_flux.T),
                        norm=norm, vmin=vmin, vmax=vmax, cmap=cmap)
        ax[1].set_aspect('equal')
        ax[1].set_xlabel(f'Longitude [{(self.binning_geometry.lon_axis.unit).to_string()}]')
        ax[1].set_ylabel(f'Latitude [{(self.binning_geometry.lat_axis.unit).to_string()}]')
        ax[1].invert_xaxis()
        plt.colorbar(mappable=pcm, 
                     ax=ax[1],)




        pcm = ax[2].pcolormesh(self.binning_geometry.energy_axis.value, 
                        self.binning_geometry.lon_axis.value, 
                        np.exp(log_integrated_mix_flux.T),
                        norm=norm, vmin=vmin, vmax=vmax, cmap=cmap)
        ax[2].set_xscale('log')
        ax[2].set_ylabel(f'Longitude [{(self.binning_geometry.lon_axis.unit).to_string()}]')
        ax[2].set_xlabel(f'Energy [{self.binning_geometry.energy_axis.unit.to_string()}]')
        plt.colorbar(mappable=pcm, 
                     ax=ax[2])


        plt.tight_layout()
        return fig, ax
