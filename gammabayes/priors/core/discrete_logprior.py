from scipy.special import logsumexp
import numpy as np
from gammabayes.samplers import  integral_inverse_transform_sampler
from gammabayes.utils import iterate_logspace_integration, construct_log_dx_mesh, update_with_defaults
import matplotlib.pyplot as plt
import warnings

class discrete_logprior(object):
    
    def __init__(self, name: str='[None]', 
                 inputunit: str=None, 
                 logfunction: callable=None, 
                 log_mesh_efficient_func: callable = None,
                 axes: tuple[np.ndarray] | None = None, 
                 axes_names: list[str] | tuple[str] = ['None'], 
                 default_spectral_parameters: dict = {},  
                 default_spatial_parameters: dict = {},  
                 iterative_logspace_integrator: callable = iterate_logspace_integration,
                 ):
        """Initialise a discrete_logprior class instance.

        Args:
            name (str, optional): A string representing the name of the 
                instance. Defaults to '[None]'.

            inputunit (list, optional): A list containing representations of the 
                units for each of the axes within axes argument. Defaults to None.

            logfunction (function): A function that outputs the log prior values 
                with input format of 
                logfunc(axis_1_val, 
                        axis_2_val,
                        ...,
                        axis_n_val, 
                        hyperparameter_value_1, 
                        hyperparameter_value_2, 
                        ...). 

            axes (tuple): A tuple of the axes that the discrete prior is 
                defined/normalised along. Generally presumed to be energy and sky position axes.

            axes_names (list, optional): A list of strings for the names of 
                the axes. Defaults to '[None]'.

            default_spectral_parameters (dict, optional): Default dictionary 
                of the parameters for spectral factor of the prior if needed. 
                Defaults to {}.

            default_spatial_parameters (dict, optional): Default dictionary 
                of the parameters for spatial factor of the prior if needed. 
                Defaults to {}.

            iterative_logspace_integrator (callable, optional): Integration
            method used for normalisation. Defaults to iterate_logspace_integration.
        """
        self.name = name
        self.inputunit = inputunit
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
        """Dunder method for what is the output when `print` is used on a class 
            instance.

        Returns:
            str: A string containing a rough description of the class instance.
        """
        string_text = 'discrete log prior class\n'
        string_text = string_text+'-'*(len(string_text)+3)+'\n'
        string_text = string_text+f'name = {self.name}\n'
        string_text = string_text+f'logfunction type is {self.logfunction}\n'
        string_text = string_text+f'input units of {self.inputunit}\n'
        string_text = string_text+f'over axes {self.axes_names}\n'        
        return string_text
    
    
    def __call__(self, *args, **kwargs)  -> np.ndarray | float:
        """Dunder method to be able to use the class in the same method 
        as the logfunction input.

        Returns:
            np.ndarray | float: Output of the logfunction for the given inputs.
        """
            
        return self.logfunction(*args, **kwargs)

    
    def normalisation(self, log_prior_values: np.ndarray = None, 
                      spectral_parameters: dict = {}, 
                      spatial_parameters: dict = {}) -> np.ndarray | float:
        """Return the integrated value of the prior for a given hyperparameter 
        over the default axes

        Args:
            hyperparametervalues (tuple, optional): Tuple of the hyperparameters 
            for the prior. Defaults to an empty list.

        Returns:
            float: the integrated value of the prior for a given hyperparameter 
        over the default axes
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
        log_prior_norms = self.logspace_integrator(logy=np.squeeze(log_prior_values), axes=self.axes)

        return log_prior_norms
    
    
    
    def sample(self, 
               numsamples: int, 
               log_prior_values: np.ndarray = None, 
               spectral_parameters: dict = None, 
               spatial_parameters: dict = None)  -> np.ndarray:
        """Returns the specified number of samples weighted by the prior 
            distribution.

        Returns the specified number of samples for the prior with the use of 
            inverse transform sampling on a discrete grid for the specified axes 
            and hyperparameters. If either are not given then the relevant 
            default is used.

        Args:
            numsamples (int): Number of wanted samples
            logpriorvalues (np.ndarray, optional): The matrix of log prior 
                values to sample, if none given one will be constructed. 

        Returns:
            np.ndarray: A numpy array containing the sampled axis values in 
                order given when generating class instance or direct input in 
                the axes argument. If 3 axes given the np.ndarray will have 
                shape (3,numsamples,).
        """
        if spectral_parameters is None:
            spectral_parameters = self.default_spectral_parameters
        if spatial_parameters is None:
            spatial_parameters = self.default_spatial_parameters

        update_with_defaults(spectral_parameters, self.default_spectral_parameters)
        update_with_defaults(spatial_parameters, self.default_spatial_parameters)
        
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
            
                
            return np.asarray(simvals)
        else:
            return  np.asarray([np.asarray([]) for idx in range(self.num_axes)])
    
    def construct_prior_array(self, 
                              spectral_parameters: dict = {}, 
                              spatial_parameters: dict = {}, 
                              normalise: bool = False,)  -> np.ndarray:
        """Construct a matrix of log prior values for input hyperparameters.

        For the input hyperparameters, if none given then the defaults are used, 
        a matrix of the log of the prior probability values for all the 
        combinations of axes values in the shape of 
        (axis_1_shape, axis_2_shape, ..., axis_n_shape) for n axes is returned.

        Args:
            hyperparameters (tuple, optional): A tuple containing the set of 
                hyperparameters for the prior that will be used. Defaults to 
                None.

            normalise (bool, optional): A bool value that if True normalises 
                the output prior with respect to the axes. Defaults to False. 

        Returns:
            log_prior_matrix (np.ndarray): A matrix containg the log prior 
                values for the input hyperparameters over the given axes
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
            normalisation = self.normalisation(log_prior_values = outputarray)

            print(normalisation.shape)
            normalisation = np.where(np.isneginf(normalisation), 0, normalisation)
            outputarray = outputarray - normalisation

            normalisation = self.normalisation(log_prior_values = outputarray)
            normalisation = np.where(np.isneginf(normalisation), 0, normalisation)
            outputarray = outputarray - normalisation

             
        return outputarray

        
