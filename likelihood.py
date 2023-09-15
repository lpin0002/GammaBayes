from scipy.special import logsumexp
import numpy as np
from inverse_transform_sampling import inverse_transform_sampling



class discrete_loglikelihood(object):
    
    def __init__(self, name='[None]', inputunit=None, logfunction=None, axes=None, 
                 dependent_axes=None, axes_names='[None]', dependent_axes_names='[None]',
                 logjacob = 0):
        self.name = name
        self.inputunit = inputunit
        self.logfunction = logfunction
        self.axes_names = axes_names
        self.axes = axes
        self.dependent_axes_names = dependent_axes_names
        self.dependent_axes = dependent_axes
        self.logjacob = logjacob
        if np.array(axes).ndim==1 or np.array(dependent_axes).ndim==1:
            if np.array(axes).ndim==1 and np.array(dependent_axes).ndim==1:
                print('beep')
                self.axes_dim = 1
                self.dependent_axes_dim = 1

                self.axes_mesh = np.meshgrid(axes, dependent_axes)
            elif np.array(axes).ndim==1:
                print('boop')

                self.axes_dim = 1
                self.axes_mesh = np.meshgrid(axes, *dependent_axes)
            else:
                print('bopp')
                print(np.array(axes).ndim)
                self.axes_dim = len(axes)
                self.dependent_axes_dim = 1

                self.axes_mesh = np.meshgrid(*axes, dependent_axes)
        else:
            print('beeeep')
            self.axes_dim = len(axes)
            print(f'Number of data dimensions {self.axes_dim}')
            self.axes_mesh = np.meshgrid(*axes, *dependent_axes)
        
        
    def __call__(self, *inputs):
        return self.logfunction(*inputs)
    
    
    
    def __repr__(self):
        string_text = 'discrete log likelihood class\n'
        string_text = string_text+'-'*(len(string_text)+3)+'\n'
        string_text = string_text+f'name = {self.name}\n'
        string_text = string_text+f'logfunction type is {self.logfunction}\n'
        string_text = string_text+f'input units of {self.inputunit}\n'
        string_text = string_text+f'over axes {self.axes_names}\n'
        string_text = string_text+f'with dependent axes {self.dependent_axes_names}\n'
        
        return string_text
    
    

    
    def normalisation(self):
        return logsumexp(self.__call__(*self.axes_mesh), axis=tuple(np.arange(self.axes.ndim))+logjacob)
    
    
    def sample(self, dependentvalues, numsamples):
        if self.axes_dim!=1:
            inputmesh = np.meshgrid(*self.axes, *dependentvalues, indexing='ij')
        else:
            inputmesh = np.meshgrid(self.axes, *dependentvalues, indexing='ij')

        
        loglikevals = np.squeeze(self.__call__(*inputmesh))+self.logjacob
        
        sampled_indices = inverse_transform_sampling(loglikevals.flatten(), numsamples)
        
        if self.axes_dim!=1:
            reshaped_simulated_indices = np.unravel_index(sampled_indices,tuple(len(arr) for arr in self.axes))
            simvals = []  
            for axis, axis_sim_index in zip(self.axes,reshaped_simulated_indices):
                simvals.append(axis[axis_sim_index])
        else:
            reshaped_simulated_indices = np.unravel_index(sampled_indices,self.axes.shape)
            simvals = self.axes[reshaped_simulated_indices]
            
        return np.array(simvals)
        
        
        
    def marginalise(self, logprior, datapoint=None, loglikelihoodvalues=None, loglikelihoodnormalisation=None, logpriorarray=None, logjacobian=0):
        
        
        
        
        if loglikelihoodvalues is None:
            if loglikelihoodnormalisation is None:
                raise UserWarning("Presuming that the loglikelihood function is normalised with respect to given axes")
            loglikelihoodvalues = self.__call__(np.meshgrid(datapoint, self.dependent_axes))
            
        
        
        if logpriorarray is not None:
            if logpriorarray.ndim==self.dependent_axes_dim:
                
                return logsumexp(logpriorarray+loglikelihoodvalues+logjacobian, axis=tuple(np.flip(-np.arange(self.dependent_axes_dim)-1)))
            
            else:
                marginalised_values = []
                for single_logpriorarray in logpriorarray:
                    marginalised_values.append(logsumexp(single_logpriorarray+loglikelihoodvalues+logjacobian, axis=tuple(np.flip(-np.arange(self.dependent_axes_dim)-1))))
                    
                return marginalised_values
