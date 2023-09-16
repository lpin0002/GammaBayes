from scipy.special import logsumexp
import numpy as np
from inverse_transform_sampling import inverse_transform_sampling
import matplotlib.pyplot as plt

class discrete_logprior(object):
    
    def __init__(self, name='[None]', 
                 inputunit=None, logfunction=None, 
                 axes=None, axes_names='[None]', 
                 hyperparameter_axes=None, hyperparameter_names='[None]',
                 default_hyperparameter_values=None, 
                 logjacob=0):
        
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


        if type(inputs)!=tuple:
            inputs = (inputs,)
            
        if hyperparameters != (None,):
            return self.logfunction(*inputs, *hyperparameters)
        else:
            return self.logfunction(*inputs)

    
    

    
    def normalisation(self, hyperparametervalues=None):
        return logsumexp(self.logfunction(self.axes_mesh, hyperparametervalues)+self.logjacob, axis=tuple(np.arange(self.axes.ndim)))
    
    
    
    def sample(self, numsamples, logpriorvalues=None):
        
        
        if logpriorvalues is None:
            try:
                logpriorvalues = self.logfunction(*self.input_values_mesh)
            except:
                logpriorvalues = self.logfunction((*self.axes_mesh,))
    
        if type(logpriorvalues)!=np.ndarray:
            logpriorvalues = np.array(logpriorvalues)
            
        logpriorvalues_withlogjacob = logpriorvalues+self.logjacob
            
        
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
    
    def construct_prior_array(self, hyperparameters=None):
        
        if hyperparameters is None:
            hyperparameters = self.default_hyperparameter_values
            
        inputmesh = np.meshgrid(*self.axes,  *hyperparameters)        
        return self.logfunction(*inputmesh)
