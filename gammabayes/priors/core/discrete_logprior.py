from scipy.special import logsumexp
import numpy as np
from gammabayes.samplers import  integral_inverse_transform_sampler
from gammabayes.utils import logspace_simpson
import matplotlib.pyplot as plt

class discrete_logprior(object):
    
    def __init__(self, name='[None]', 
                 inputunit=None, logfunction=None, 
                 axes=None, axes_names='[None]', 
                 hyperparameter_axes=None, hyperparameter_names='[None]',
                 default_hyperparameter_values=None, 
                 logjacob=0):
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
                defined/normalised along. Presumed to be energy and sky position axes.

            axes_names (list, optional): A list of strings for the names of 
                the axes. Defaults to '[None]'.

            hyperparameter_axes (tuple, optional): A tuple of the sets 
                default hyperparameter axis vals for the priors to be 
                    evaluated at. If there are two priors for example, there 
                    would be two tuples containing the tuple of hyperparameter 
                    axes for each respective prior. Defaults to None.

            hyperparameter_names (list, optional): A list containing the 
                names of the hyperparameter axes. Defaults to '[None]'.

            default_hyperparameter_values (tuple, optional): Default values 
                of the hyperparameters for the prior if needed. Defaults to 
                    None.

            logjacob (int, optional): Natural log of the jacobian needed for 
                normalisation, if axes are of size (axis_1,), (axis_2,), ..., 
                (axis_n,). Defaults to 0.
        """
        self.name = name
        self.inputunit = inputunit
        self.logfunction = logfunction
        self.axes_names = axes_names
        
        self.hyperparameter_axes = hyperparameter_axes
        self.hyperparameter_names = hyperparameter_names
        self.num_axes = len(axes)
        self.logjacob = logjacob
        if self.num_axes==1:
            self.axes_mesh = (axes,)
            self.axes = (axes,)
        else:
            self.axes = axes
            self.axes_mesh = (*np.meshgrid(*axes, indexing='ij'),)
            
        if default_hyperparameter_values is None:
            self.default_hyperparameter_values = None
            self.input_values_mesh = np.meshgrid(*self.axes, indexing='ij')

        else:
            self.default_hyperparameter_values = (*default_hyperparameter_values,)
            self.input_values_mesh = np.meshgrid(*self.axes, *self.default_hyperparameter_values, indexing='ij')

            

            
    
    
    
    def __repr__(self):
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
        string_text = string_text+f'with hyperparameter(s) {self.hyperparameter_names}\n'        
        return string_text
    
    
    def __call__(self, *args, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
            
        return self.logfunction(*args, **kwargs)


    
    

    
    def normalisation(self, log_prior_values=None, hyperparametervalues=None):
        """Return the integrated value of the prior for a given hyperparameter 
        over the default axes

        Args:
            hyperparametervalues (tuple, optional): Tuple of the hyperparameters 
            for the prior. Defaults to None.

        Returns:
            float: the integrated value of the prior for a given hyperparameter 
        over the default axes
        """

        if (log_prior_values is None) and (hyperparametervalues is None):
            log_prior_values = self.logfunction(self.axes_mesh, **self.default_hyperparameter_values)
        elif (log_prior_values is None) and not(hyperparametervalues is None):
            log_prior_values = self.logfunction(self.axes_mesh, **hyperparametervalues)

        log_prior_norms = log_prior_values
        for axis in self.axes:
            log_prior_norms = logspace_simpson(logy=log_prior_norms, x=axis, axis=0)

        return log_prior_norms
    
    
    
    def sample(self, numsamples, logpriorvalues=None):
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

            logpriorvalues (array like): Log probability values for a different 
            prior than the default

        Returns:
            np.ndarray: A numpy array containing the sampled axis values in 
                order given when generating class instance or direct input in 
                the axes argument. If 3 axes given the np.ndarray will have 
                shape (3,numsamples,).
        """
        if numsamples>0:
            if logpriorvalues is None:
                logpriorvalues = self.logfunction(*self.input_values_mesh)
        
            if type(logpriorvalues)!=np.ndarray:
                logpriorvalues = np.array(logpriorvalues)
                
                
                
            # This code is presuming a large number of events. This can cause a lot of numerical instability issues down the line 
                # of a hierarchical models (especially without the use of samplers which is currently the case for this code)
                # So we will double check the normalisation
            logpriorvalues = np.squeeze(logpriorvalues) - self.normalisation(logpriorvalues)
            logpriorvalues = np.squeeze(logpriorvalues) - self.normalisation(logpriorvalues)
            logpriorvalues = np.squeeze(logpriorvalues) - self.normalisation(logpriorvalues)
                        
            simvals = integral_inverse_transform_sampler(logpriorvalues, axes=self.axes, 
                                                Nsamples=numsamples, logjacob=self.logjacob)
            
            # simulatedindices = inverse_transform_sampler(logpriorvalues_flattened, Nsamples=numsamples)
            
            
            # reshaped_simulated_indices = np.unravel_index(simulatedindices,logpriorvalues.shape)
            
            
            # if self.num_axes==1:
            #     simvals = self.axes[reshaped_simulated_indices]
            # else:
            #     simvals = []  
            #     for axis, axis_sim_index in zip(self.axes,reshaped_simulated_indices):
            #         simvals.append(axis[axis_sim_index])
                
            return np.array(simvals)
        else:
            return  np.array([np.array([]) for idx in range(self.num_axes)])
    
    def construct_prior_array(self, hyperparameters=None, normalise=False):
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
        
        if hyperparameters is None:
            hyperparameters = self.default_hyperparameter_values

        try:
            inputmesh = np.meshgrid(*self.axes,*hyperparameters, indexing='ij') 
            outputarray = self.logfunction(*inputmesh)            

        except:
            inputmesh = np.meshgrid(*self.axes, indexing='ij') 
            outputarray = self.logfunction(*inputmesh)            


        # This is left as an option to decrease computation time
        if normalise:
            outputarray = outputarray - self.normalisation(outputarray)
            outputarray = outputarray - self.normalisation(outputarray)
            outputarray = outputarray - self.normalisation(outputarray)
             
        return outputarray

        
