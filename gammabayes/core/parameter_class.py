import warnings, numpy as np
import copy

class Parameter(dict):
    def __init__(self, initial_data=None):
        data_copy = copy.deepcopy(initial_data) if initial_data is not None else {}
        
        super().__init__(data_copy)

        if not('name' in self):
            self['name'] = 'UnnamedParameter'

        if not('scaling' in self):
            self['scaling'] = 'linear'

        if self['scaling'] not in ['linear', 'log10']:
            warnings.warn("Scaling typically must be 'linear' or 'log10'. ")

        if not('discrete' in self):
            self['discrete'] = True

        if not('default_value' in self):
            self['default_value'] = 1.0

        self['default_value'] = float(self['default_value'])

        if not('bounds' in self):
            warnings.warn("No bounds given. Defaulting to 0. and 1.")
            self['bounds'] = [0., 1.]
        
        if self['bounds']=='event_dynamic':
            self['num_events'] = float(self['num_events'])


            self['dynamic_multiplier'] = float(self['dynamic_multiplier'])

            if self['scaling']=='linear':
                self['bounds'] = [self['default_value']-self['dynamic_multiplier']/np.sqrt(self['num_events']), 
                                  self['default_value']+self['dynamic_multiplier']/np.sqrt(self['num_events'])]
            else:
                self['bounds'] = [10**(np.log10(self['default_value'])-np.log10(self['dynamic_multiplier'])/np.sqrt(self['num_events'])), 
                                  10**(np.log10(self['default_value'])+np.log10(self['dynamic_multiplier'])/np.sqrt(self['num_events']))]
        
        for idx, bound in enumerate(self['bounds']):
            self['bounds'][idx] = float(bound)

        if self['discrete']:

            if not('bins' in self) and (self['scaling'] == 'linear'):  

                # Defaults to 10 per unit plus one so that if the bounds are 
                    # 0.0 --> 1.0 you get 
                    # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
                    # __not__
                    # 0.0, 0.111..., 0.222..., 0.333..., 0.444..., 0.555..., etc
                self['bins'] = int(np.round(np.diff(self['bounds'])*10+1))

            elif not('bins' in self) and (self['scaling'] == 'log10'):  

                # Defaults to 10 per unit plus one to mimic the linear behaviour
                    # but in log10-space
                self['bins'] = int(np.round(np.diff(np.log10(self['bounds']))*10+1))


            if not('axis' in self):
                if type(self['bins'])==int:
                    if self['scaling']=='linear':
                        self['axis'] = np.linspace(*self['bounds'], self['bins'])
                    else:
                        self['axis'] = np.logspace(np.log10(self['bounds'][0]), 
                                                   np.log10(self['bounds'][1]), 
                                                   self['bins'])
                        
                
            self['transform_scale'] = self['bins']
            if 'custom_scaling' in self:
                self.transform = self['custom_parameter_transform']
            else:
                self.transform = self.discrete_parameter_transform
        else:
            if 'custom_scaling' in self:
                self.transform = self['custom_parameter_transform']
            else:
                if self['scaling']=='linear':
                    self['transform_scale'] = float(np.diff(self['bounds']))
                    self.transform = self.continuous_parameter_transform_linear_scaling
                else:
                    self['transform_scale'] = float(np.diff(np.log10(self['bounds'])))
                    self.transform = self.continuous_parameter_transform_log10_scaling

        if not('prior_id' in self):
            self['prior_id'] = 0


        if not('likelihood_id' in self):
            self['likelihood_id'] = 0

        if not('parameter_type' in self):
            self['parameter_type'] = 'spectral_parameters'


    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'Parameter' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value
        super().__setattr__(key, value)


    def discrete_parameter_transform(self, u):

        scaled_value = u * self.transform_scale
        index = int(np.floor(scaled_value))
        # Ensure index is within bounds
        index = max(0, min(index, self.bins - 1))
        u = self.axis[index]

        return u
    
    def continuous_parameter_transform_linear_scaling(self, u):

        u = u * self.transform_scale + self.bounds[0]

        return u

    
    def continuous_parameter_transform_log10_scaling(self, u):

        u = 10**(u * self.transform_scale + np.log10(self.bounds[0]))

        return u        
    

    