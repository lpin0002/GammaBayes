from scipy.special import logsumexp
import numpy as np
from GammaBayes.gammabayes.utils.inverse_transform_sampling import inverse_transform_sampling
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
                defined/normalised along.

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
            self.default_hyperparameter_values = (None,)
            self.input_values_mesh = np.meshgrid(*self.axes, indexing='ij')

        else:
            self.default_hyperparameter_values = (*default_hyperparameter_values,)
            print(self.default_hyperparameter_values)
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
    
    
    def __call__(self, inputs, hyperparameters=None):
        """A dunder method to allow a class instance to be used as a function.

        Args:
            inputs (_type_): _description_
            hyperparameters (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if hyperparameters is None:
            hyperparameters = self.default_hyperparameter_values


        if type(inputs)!=tuple:
            inputs = (inputs,)
            
        if hyperparameters != (None,):
            return self.logfunction(*inputs, *hyperparameters)
        else:
            return self.logfunction(*inputs)

    
    

    
    def normalisation(self, hyperparametervalues=None):
        return logsumexp(self.logfunction(*self.axes_mesh, *hyperparametervalues)+self.logjacob, axis=tuple(np.arange(self.axes.ndim)))
    
    
    
    def sample(self, numsamples, logpriorvalues=None, axes=None, hyperparameters=None):
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

            axes (tuple, optional): A tuple containing the axes for the prior to 
                be evaluated at to generate the matrix, if none given default of 
                instance will be used.

            hyperparameters (tuple, optional): A tuple containing the 
                hyperparameters for the prior to construct the matrix, if none 
                given default of instance will be used.

        Returns:
            np.ndarray: A numpy array containing the sampled axis values in 
                order given when generating class instance or direct input in 
                the axes argument. If 3 axes given the np.ndarray will have 
                shape (3,numsamples,).
        """
        if (axes is None) and (hyperparameters is None):
            inputmesh = self.input_values_mesh
        elif (axes is None) and not(hyperparameters is None):
            inputmesh = np.meshgrid(*self.axes, *hyperparameters, indexing='ij')
        elif not(axes is None) and (hyperparameters is None):
            inputmesh = np.meshgrid(*axes, *self.default_hyperparameter_values, indexing='ij')
        else:
            inputmesh = np.meshgrid(*axes, *hyperparameters, indexing='ij')


        
        if logpriorvalues is None:
            logpriorvalues = self.logfunction(*inputmesh)
    
        if type(logpriorvalues)!=np.ndarray:
            logpriorvalues = np.array(logpriorvalues)
            
        logpriorvalues_withlogjacob = np.squeeze(logpriorvalues)+self.logjacob
            
            
        # This code is presuming a large number of events. This can cause a lot of numerical instability issues down the line 
            # of a hierarchical models (especially without the use of samplers which is currently the case for this code)
            # So we will double check the normalisation
        logpriorvalues_withlogjacob = logpriorvalues_withlogjacob - logsumexp(logpriorvalues_withlogjacob)
        logpriorvalues_withlogjacob = logpriorvalues_withlogjacob - logsumexp(logpriorvalues_withlogjacob)
        
        logpriorvalues_flattened = logpriorvalues_withlogjacob.flatten()
        
        
        simulatedindices = inverse_transform_sampling(logpriorvalues_flattened, Nsamples=numsamples)
        
        
        reshaped_simulated_indices = np.unravel_index(simulatedindices,logpriorvalues.shape)
        
        
        if self.num_axes==1:
            simvals = self.axes[reshaped_simulated_indices]
        else:
            simvals = []  
            for axis, axis_sim_index in zip(self.axes,reshaped_simulated_indices):
                simvals.append(axis[axis_sim_index])
            
        return np.array(simvals)
    

    
    def construct_prior_array(self, hyperparameters=None, normalise=False, axes=None):
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

            axes (tuple, optional): A tuple containing numpy arrays representing 
                the discrete values at which the prior can be evaluated. 
                Defaults to None.

        Returns:
            log_prior_matrix (np.ndarray): A matrix containg the log prior 
                values for the input hyperparameters over the given axes
        """

        if axes is None:
            axes=self.axes
        
        if hyperparameters is None:
            hyperparameters = self.default_hyperparameter_values
        try:
            inputmesh = np.meshgrid(*axes,  *hyperparameters, indexing='ij') 
            outputarray = self.logfunction(*inputmesh)
            
        except:
            inputmesh = np.meshgrid(*axes, indexing='ij')  
            outputarray = self.logfunction(*inputmesh)
            

        # This is left as an option to decrease computation time
        if normalise:
            outputarray = outputarray - logsumexp(outputarray.reshape(self.logjacob.shape)+self.logjacob, axis=(*np.arange(len(axes)),))
            outputarray = outputarray - logsumexp(outputarray.reshape(self.logjacob.shape)+self.logjacob, axis=(*np.arange(len(axes)),))
             
        return outputarray

        
