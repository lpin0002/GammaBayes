from scipy.special import logsumexp
import numpy as np
from ..utils.inverse_transform_sampling import inverse_transform_sampling
from matplotlib import pyplot as plt


class discrete_loglikelihood(object):
    
    def __init__(self, name='[None]', inputunit=None, logfunction=None, axes=None, 
                 dependent_axes=None, axes_names='[None]', dependent_axes_names='[None]',
                 logjacob = 0):
        """_summary_

        Args:
            name (str, optional): _description_. Defaults to '[None]'.

            inputunit (_type_, optional): _description_. Defaults to None.

            logfunction (_type_, optional): _description_. Defaults to None.

            axes (_type_, optional): _description_. Defaults to None.

            dependent_axes (_type_, optional): _description_. Defaults to None.

            axes_names (str, optional): _description_. Defaults to '[None]'.

            dependent_axes_names (str, optional): _description_. Defaults to '[None]'.
            
            logjacob (int, optional): _description_. Defaults to 0.
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

        Returns:
            _type_: _description_
        """
        return self.logfunction(*inputs)
    
    
    
    def __repr__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        string_text = 'discrete log likelihood class\n'
        string_text = string_text+'-'*(len(string_text)+3)+'\n'
        string_text = string_text+f'name = {self.name}\n'
        string_text = string_text+f'logfunction type is {self.logfunction}\n'
        string_text = string_text+f'input units of {self.inputunit}\n'
        string_text = string_text+f'over axes {self.axes_names}\n'
        string_text = string_text+f'with dependent axes {self.dependent_axes_names}\n'
        
        return string_text
    
    
    def sample(self, dependentvalues, numsamples):
        """_summary_

        Args:
            dependentvalues (_type_): _description_
            numsamples (_type_): _description_

        Returns:
            _type_: _description_
        """
        inputmesh = np.meshgrid(*self.axes, *dependentvalues, indexing='ij')        

        
        loglikevalswithlogjacob = np.squeeze(self.__call__(*(input.flatten() for input in inputmesh)).reshape(inputmesh[0].shape))+self.logjacob
        
        loglikevalswithlogjacob = loglikevalswithlogjacob - logsumexp(loglikevalswithlogjacob, axis=(*np.arange(self.axes_dim),))
        loglikevalswithlogjacob = loglikevalswithlogjacob - logsumexp(loglikevalswithlogjacob, axis=(*np.arange(self.axes_dim),))

        
        sampled_indices = np.squeeze(inverse_transform_sampling(loglikevalswithlogjacob.flatten(), numsamples))
            
        reshaped_simulated_indices = np.unravel_index(sampled_indices, self.axes_shape)
        
        simvals = []  
        for axis, axis_sim_index in zip(self.axes,reshaped_simulated_indices):
            simvals.append(axis[axis_sim_index])
       
            
        return np.array(simvals).T
        
        
        

