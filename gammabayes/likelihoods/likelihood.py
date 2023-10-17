from scipy.special import logsumexp
import numpy as np
from ..utils.inverse_transform_sampling import inverse_transform_sampling
from matplotlib import pyplot as plt


class discrete_loglikelihood(object):
    
    def __init__(self, name='[None]', inputunit=None, logfunction=None, axes=None, 
                 dependent_axes=None, axes_names=['None'], dependent_axes_names=['None'],
                 logjacob = 0):
        """Initialise a discrete_loglikelihood class instance.

        Args:
            name (str, optional): Name given to an instance of the class. 
                Defaults to '[None]'.

            inputunit (list or tuple, optional): A list or tuple containing 
                representations of the units of the 'axes' arguments. 
                Defaults to None.

            logfunction (func): A function that outputs the log 
                likelihood values for the relevant axes and dependent axes. 

            axes (tuple, optional): A tuple of the discrete value axes on which
            the likelihood can/will be evaluated that the likelihood is/would 
            be normalised with respect to. Defaults to None.

            dependent_axes (tuple, optional): A tuple of the discrete value 
            axes on which the likelihood is dependent on that the likelihood is
            not normalised with respect to. Defaults to None.

            axes_names (list, optional): A list of the names of the axes within
            the axes argument. 
            Defaults to ['None'].

            dependent_axes_names (list, optional): A list of the names of the 
            axes within the dependent axesaxes argument. 
            Defaults to ['None'].

            logjacob (float or np.ndarray, optional): The jacobian used for normalisation,
            if the input axes are of shapes (m_1,), (m_2,), ..., (m_n) then the jacobian 
            is either a float or np.ndarray of shape (m_1, m_2, ..., m_n,). 
            Defaults to 0.
        """

        self.name = name
        self.inputunit = inputunit
        self.logfunction = logfunction
        self.axes_names = axes_names
        self.axes = axes
        print(f'Number of input dimensions {len(self.axes)}')
        self.dependent_axes_names = dependent_axes_names
        self.dependent_axes = dependent_axes
        self.logjacob = logjacob
        if len(self.axes)==1 or len(self.dependent_axes)==1:
            if len(self.axes)==1 and len(self.dependent_axes)==1:
                self.axes_dim = 1
                self.dependent_axes_dim = 1
                self.axes_shape = self.axes[0].shape

            elif len(self.axes)==1:
                self.axes_shape = self.axes[0].shape
                self.axes_dim = 1
                self.dependent_axes_dim = len(self.dependent_axes)

            else:
                print(np.array(axes).ndim)
                self.axes_dim = len(axes)
                # If it is not done this way self.axes_shape gives out a generator object location instead :(
                self.axes_shape = (*(axis.shape[0] for axis in self.axes),)
                
                self.dependent_axes_dim = 1
        else:
            self.axes_dim = len(axes)
            
            # If it is not done this way self.axes_shape gives out a generator object location instead :(
            self.axes_shape = (*(axis.shape[0] for axis in self.axes),)

            self.dependent_axes_dim = len(self.dependent_axes)
                    
        
    def __call__(self, *inputs):
        """_summary_
        Args:
            input: Inputs in the same format as would be used for the 
                logfunction used in the class.

        Returns:
            float or np.ndarray: The log-likelihood values for the given inputs.
        """
        return self.logfunction(*inputs)
    
    
    
    def __repr__(self):
        """Dunder method for what is the output when `print` is used on a class 
            instance.

        Returns:
            str: A string containing a rough description of the class instance.
        """
        string_text = 'discrete log likelihood class\n'
        string_text = string_text+'-'*(len(string_text)+3)+'\n'
        string_text = string_text+f'name = {self.name}\n'
        string_text = string_text+f'logfunction type is {self.logfunction}\n'
        string_text = string_text+f'input units of {self.inputunit}\n'
        string_text = string_text+f'over axes {self.axes_names}\n'
        string_text = string_text+f'with dependent axes {self.dependent_axes_names}\n'
        
        return string_text
    
    
    def sample(self, dependentvalues, numsamples=1, axes=None, logjacob=None):
        """Returns the specified number of samples weighted by the likelihood
            distribution.

        Args:
            dependentvalues (tuple): A tuple of the dependent values to create
                a matrix of probabilities values.

            numsamples (int): Number of samples required.
            Defaults to 1.

            axes (tuple, optional): Axes to be used instead of default is 
                required. 

            logjacob (float or np.ndarray, optional): Natural log of the jacobian
                to be used if using different axes or default logjacobian is 
                incorrect.

        Returns:
            np.ndarray: Array of sampled values.
        """
        if axes is None:
            inputmesh = np.meshgrid(*self.axes, *dependentvalues, indexing='ij') 
            axes = self.axes
            axes_shape = self.axes_shape
            axes_dim = self.axes_dim
        else:
            inputmesh = np.meshgrid(axes, *dependentvalues, indexing='ij') 
            axes_shape = (*(axis.shape[0] for axis in axes),)
            axes_dim =len(axes)

        if logjacob is None:
            logjacob = self.logjacob



        
        loglikevalswithlogjacob = np.squeeze(self.__call__(*(input.flatten() for input in inputmesh)).reshape(inputmesh[0].shape))+logjacob
        
        loglikevalswithlogjacob = loglikevalswithlogjacob - logsumexp(loglikevalswithlogjacob, axis=(*np.arange(axes_dim),))
        loglikevalswithlogjacob = loglikevalswithlogjacob - logsumexp(loglikevalswithlogjacob, axis=(*np.arange(axes_dim),))

        
        sampled_indices = np.squeeze(inverse_transform_sampling(loglikevalswithlogjacob.flatten(), numsamples))
            
        reshaped_simulated_indices = np.unravel_index(sampled_indices, axes_shape)
        
        simvals = []  
        for axis, axis_sim_index in zip(axes,reshaped_simulated_indices):
            simvals.append(axis[axis_sim_index])
       
            
        return np.array(simvals).T
        
        
        

