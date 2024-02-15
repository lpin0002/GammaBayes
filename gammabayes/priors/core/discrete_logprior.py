from scipy.special import logsumexp
import numpy as np
from gammabayes.samplers import  integral_inverse_transform_sampler
from gammabayes.utils import iterate_logspace_integration, construct_log_dx_mesh, update_with_defaults
from gammabayes.core import EventData
import matplotlib.pyplot as plt
import warnings, logging
import h5py, pickle

class DiscreteLogPrior(object):
    
    def __init__(self, name: str='[None]', 
                 inputunits: str=None, 
                 logfunction: callable=None, 
                 log_mesh_efficient_func: callable = None,
                 axes: tuple[np.ndarray] | None = None, 
                 axes_names: list[str] | tuple[str] = ['None'], 
                 default_spectral_parameters: dict = {},  
                 default_spatial_parameters: dict = {},  
                 iterative_logspace_integrator: callable = iterate_logspace_integration,
                 ):
        """
        Initializes a DiscreteLogPrior object, which represents a discrete log prior distribution.

        Parameters:
        - name (str, optional): Name of the instance. Defaults to '[None]'.
        
        - inputunits (str, optional): Unit of the input values for the axes. Defaults to None.
        
        - logfunction (callable, optional): A function that calculates log prior values given axes values and hyperparameters.
          The function should accept arguments in the order of axis values followed by hyperparameter values.
        
        - log_mesh_efficient_func (callable, optional): A function for efficient computation of log prior values on a mesh grid.
          If not provided, a warning is raised about the lack of efficient computation.
        
        - axes (tuple[np.ndarray], optional): A tuple containing np.ndarray objects for each axis over which the prior is defined.
          Generally represents energy and sky position axes.
        
        - axes_names (list[str] | tuple[str], optional): Names of the axes. Defaults to ['None'].
        
        - default_spectral_parameters (dict, optional): Default spectral parameters for the prior. Defaults to an empty dict.
        
        - default_spatial_parameters (dict, optional): Default spatial parameters for the prior. Defaults to an empty dict.
        
        - iterative_logspace_integrator (callable, optional): Function used for integrations in log space. Defaults to iterate_logspace_integration.

        Note:
        - This class assumes the prior is defined in a discrete log space along specified axes.
        - The axes should correspond to physical quantities over which the prior is distributed, such as energy and sky coordinates.
        """
        self.name = name
        self.inputunits = inputunits
        self.logfunction = logfunction
        self.axes_names = axes_names
        
        self.axes = axes
        self.num_axes = len(axes)

        if not(log_mesh_efficient_func is None):
            self.efficient_exist = True
            self.log_mesh_efficient_func = log_mesh_efficient_func
        else:
            self.efficient_exist = False
            warnings.warn('No function to calculate on mesh efficiently given')

        if self.num_axes==1:
            self.axes_mesh = (axes,)
        else:
            self.axes_mesh = np.meshgrid(*axes, indexing='ij')

            
        self.default_spectral_parameters = default_spectral_parameters
        self.default_spatial_parameters = default_spatial_parameters

        self.num_spec_params = len(default_spectral_parameters)
        self.num_spat_params = len(default_spatial_parameters)


        self.logspace_integrator = iterative_logspace_integrator
            

            
    
    
    
    def __repr__(self) -> str:
        """
        String representation of the DiscreteLogPrior instance.

        Returns:
        - str: A description of the instance including its name, logfunction type, input units, and axes names.
        """
        description = f"Discrete log prior class\n{'-' * 20}\n" \
                      f"Name: {self.name}\n" \
                      f"Logfunction type: {type(self.logfunction).__name__}\n" \
                      f"Input units: {self.inputunits}\n" \
                      f"Axes: {self.axes_names}\n"
        return description
    
    
    def __call__(self, *args, **kwargs)  -> np.ndarray | float:
        """
        Allows the instance to be called like a function, passing arguments directly to the logfunction.

        Parameters:
        - *args: Arguments for the logfunction.
        
        - **kwargs: Keyword arguments for the logfunction.

        Returns:
        - np.ndarray | float: The result from the logfunction, which is the log prior value(s) for the given input(s).
        """
        return self.logfunction(*args, **kwargs)

    
    def normalisation(self, log_prior_values: np.ndarray = None, 
                      spectral_parameters: dict = {}, 
                      spatial_parameters: dict = {},
                      axisindices: list = [0,1,2]) -> np.ndarray | float:
        """
        Calculates the normalisation constant of the log prior over specified axes.

        Parameters:
        - log_prior_values (np.ndarray, optional): Pre-computed log prior values. If None, they will be computed using default or provided hyperparameters.
        - spectral_parameters (dict, optional): Spectral parameters to be used if log_prior_values is not provided. Defaults to instance's default spectral parameters.
        - spatial_parameters (dict, optional): Spatial parameters to be used if log_prior_values is not provided. Defaults to instance's default spatial parameters.
        - axisindices (list, optional): Indices of the axes over which to integrate. Defaults to [0, 1, 2].

        Returns:
        - np.ndarray | float: The normalisation constant for the log prior, either as a scalar or an array depending on the integration over multiple axes.
        """
        

        if (log_prior_values is []) | (log_prior_values is None):
            # Checks if spectral_parameters is an empty dict. If so, it sets it to the defaults
            update_with_defaults(spectral_parameters, self.default_spectral_parameters)
            update_with_defaults(spatial_parameters, self.default_spatial_parameters)

            if self.efficient_exist:
                log_prior_values = self.log_mesh_efficient_func(*self.axes, 
                                                                spectral_parameters = spectral_parameters,
                                                                spatial_parameters = spatial_parameters)
            else:
                inputmesh = np.meshgrid(*self.axes, 
                                *spectral_parameters.values(),
                                *spatial_parameters.values(), indexing='ij') 
        
                log_prior_values = self.logfunction(*inputmesh[:self.num_axes], 
                                                spectral_parameters = {hyper_key: inputmesh[self.num_axes+idx] for idx, hyper_key in enumerate(spectral_parameters.keys())}, 
                                                spatial_parameters = {hyper_key: inputmesh[self.num_axes+len(spectral_parameters)+idx] for idx, hyper_key in enumerate(spatial_parameters.keys())}
                                                )    
        log_prior_norms = self.logspace_integrator(logy=np.squeeze(log_prior_values), axes=self.axes, axisindices=axisindices)

        return log_prior_norms
    
    
    
    def sample(self, 
               numsamples: int, 
               log_prior_values: np.ndarray = None, 
               spectral_parameters: dict = None, 
               spatial_parameters: dict = None)  -> np.ndarray:
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

        numsamples = int(round(numsamples))
        
        if numsamples>0:
            if log_prior_values is None:
                if self.efficient_exist:
                    log_prior_values = self.log_mesh_efficient_func(*self.axes, 
                                                                  spectral_parameters = spectral_parameters,
                                                                  spatial_parameters = spatial_parameters)

                else:
                    inputmesh = np.meshgrid(*self.axes, 
                                    *spectral_parameters.values(),
                                    *spatial_parameters.values(), indexing='ij') 
            
                    log_prior_values = self.logfunction(*inputmesh[:self.num_axes], 
                                                    spectral_parameters = {hyper_key: inputmesh[self.num_axes+idx] for idx, hyper_key in enumerate(spectral_parameters.keys())}, 
                                                    spatial_parameters = {hyper_key: inputmesh[self.num_axes+len(spectral_parameters)+idx] for idx, hyper_key in enumerate(spatial_parameters.keys())}
                                                    )
        
            log_prior_values = np.asarray(log_prior_values)
                
                
                
            # This code is presuming a large number of events. This can cause a lot of numerical instability issues down the line 
                # of a hierarchical models (especially without the use of samplers which is currently the case for this code)
                # So we will double check the normalisation
            log_prior_values = np.squeeze(log_prior_values) - self.normalisation(log_prior_values=log_prior_values)

            logdx = construct_log_dx_mesh(self.axes)
                        
            simvals = integral_inverse_transform_sampler(log_prior_values+logdx, axes=self.axes, 
                                                Nsamples=numsamples)
            
                
            return EventData(data=np.asarray(simvals).T, 
                             energy_axis=self.axes[0], 
                             glongitude_axis=self.axes[1], 
                             glatitude_axis=self.axes[2], 
                             _source_ids=[self.name]*numsamples,
                             _true_vals = True
                             )
        else:
            return  EventData(energy_axis=self.axes[0], 
                             glongitude_axis=self.axes[1], 
                             glatitude_axis=self.axes[2], 
                             _true_vals = True
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
            outputarray = self.log_mesh_efficient_func(*self.axes, 
                                                       spectral_parameters=spectral_parameters, 
                                                       spatial_parameters=spatial_parameters)
            

        else:
            inputmesh = np.meshgrid(*self.axes, 
                                    *spectral_parameters.values(),
                                    *spatial_parameters.values(), indexing='ij') 
            
            outputarray = self.logfunction(*inputmesh[:self.num_axes], 
                                            spectral_parameters = {hyper_key: inputmesh[self.num_axes+idx] for idx, hyper_key in enumerate(spectral_parameters.keys())}, 
                                            spatial_parameters = {hyper_key: inputmesh[self.num_axes+len(spectral_parameters)+idx] for idx, hyper_key in enumerate(spatial_parameters.keys())}
                                            )
    
        if normalise:
            # Normalisation is done twice to reduce numerical instability issues
            normalisation = self.normalisation(log_prior_values = outputarray, axisindices=normalisation_axes)

            logging.info(f"normalisation.shape: {normalisation.shape}")
            normalisation = np.where(np.isinf(normalisation), 0, normalisation)
            outputarray = outputarray - normalisation

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
    def load(cls, file_name):
        if not(file_name.endswith(".pkl")):
            file_name = file_name + ".pkl"
        return  pickle.load(open(file_name,'rb'))




