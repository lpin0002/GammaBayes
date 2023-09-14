from scipy.special import logsumexp
import numpy as np
from inverse_transform_sampling import inverse_transform_sampling
import matplotlib.pyplot as plt

class discrete_logprior(object):
    
    def __init__(self, name='[None]', 
                 inputunit=None, logfunction=None, 
                 axes=None, axes_names='[None]', 
                 hyperparameter_axes=None, hyperparameter_names='[None]',
                 default_hyperparameter_values=[]):
        
        self.name = name
        self.inputunit = inputunit
        self.logfunction = logfunction
        self.axes_names = axes_names
        self.axes = axes
        self.hyperparameter_axes = hyperparameter_axes
        self.hyperparameter_names = hyperparameter_names
        self.num_axes = len(axes)
        if self.num_axes==1:
            self.axes_mesh = (axes,)
        else:
            self.axes_mesh = np.meshgrid(*axes, indexing='ij')
            
        
        self.default_hyperparameter_values = default_hyperparameter_values
            
    
    
    
    def __repr__(self):
        string_text = 'discrete log prior class\n'
        string_text = string_text+'-'*(len(string_text)+3)+'\n'
        string_text = string_text+f'name = {self.name}\n'
        string_text = string_text+f'logfunction type is {self.logfunction}\n'
        string_text = string_text+f'input units of {self.inputunit}\n'
        string_text = string_text+f'over axes {self.axes_names}\n'        
        string_text = string_text+f'with hyperparameter(s) {self.hyperparameter_names}\n'        
        return string_text
    
    
    def __call__(self, inputs, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = self.default_hyperparameter_values
            print(hyperparameters)
            
            
        return self.logfunction(inputs, hyperparameters)
    
    

    
    def normalisation(self, hyperparametervalues=None):
        return logsumexp(self.logfunction(self.axes_mesh, hyperparametervalues), axis=tuple(np.arange(self.axes.ndim)))
    
    
    
    def sample(self, numsamples, hyperparametervalues=None, logpriorvalues=None):
        if hyperparametervalues is None:
            hyperparametervalues = self.default_hyperparameter_values
        
        
        if logpriorvalues is None:
            print(*self.axes_mesh)
            logpriorvalues = self.logfunction(*self.axes_mesh, hyperparametervalues)
        
        if type(logpriorvalues)!=np.ndarray:
            logpriorvalues = np.array(logpriorvalues)
        
        logpriorvalues_flattened = logpriorvalues.flatten()
        
        simulatedindices = inverse_transform_sampling(logpriorvalues_flattened, Nsamples=numsamples)
        
        
        reshaped_simulated_indices = np.unravel_index(simulatedindices,logpriorvalues.shape)
        
        
        if self.num_axes==1:
            simvals = self.axes[reshaped_simulated_indices]
        else:
            simvals = []  
            for axis, axis_sim_index in zip(self.axes,reshaped_simulated_indices):
                simvals.append(axis[axis_sim_index])
            
        return np.array(simvals)