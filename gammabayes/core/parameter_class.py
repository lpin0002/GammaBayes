import warnings, numpy as np
import copy, h5py, pickle
from scipy.stats import uniform, loguniform

class Parameter(dict):
    """
    A class to define and manipulate parameters for simulations or models, 
    with support for linear and logarithmic scaling, discrete and continuous 
    values, and dynamic bounds based on event numbers.

    Inherits from dict, allowing it to store various attributes of parameters 
    like scaling, bounds, default values, and more.

    Attributes are dynamically adjustable and support serialization to and 
    from HDF5 files for easy sharing and storage of configurations.
    """
    def __init__(self, initial_data: dict =None, **kwargs):
        """
        Initializes a Parameter object with specified attributes and properties.

        Args:
            initial_data (dict, optional): A dictionary of initial parameter attributes.
            **kwargs: Arbitrary keyword arguments for additional attributes.

        The constructor initializes with default settings if certain attributes (scaling, 
        discrete, default_value, bounds) are not specified. It also handles the conversion 
        of bounds to numerical values and sets up the parameter transformation functions 
        based on scaling and discreteness.
        """

        if initial_data is None:
            initial_data = kwargs


            
        if type(initial_data) == dict:
            data_copy = copy.deepcopy(initial_data) if initial_data is not None else {}
            data_copy.update(kwargs)
            
            super().__init__(data_copy)

            if not('scaling' in self):
                self['scaling'] = 'linear'


            # log10 to specifically let people now that the log will be base 10
                # rather than the natural scale which is generally notated as 
                # 'log' for the rest of the repo/package/project/thingy
            if self['scaling'] not in ['linear', 'log10']:
                warnings.warn("Scaling typically must be 'linear' or 'log10'. ")


            if not('discrete' in self):
                self['discrete'] = True

            if not('default_value' in self):
                self['default_value'] = 1.0


            # A default value does not have to be given but is helpful, and would 
                # be required if one uses event dynamic bounds
            self['default_value'] = float(self['default_value'])

            # For discrete values we highly recommend you explicitly state the bounds
                # even if they are 0. and 1.
            if not('bounds' in self):
                warnings.warn("No bounds given. Defaulting to 0. and 1.")
                self['bounds'] = [0., 1.]
            
            # A specific way of setting the bounds on the parameter if a value is
                # fairly well known (or set within simulations) and this way makes
                # the bounds change with the square root of the number of events
                # (assuming CLT essentially). 
            
            if isinstance(self['bounds'], str):
                if self['bounds']=='event_dynamic':
                    self['num_events'] = float(self['num_events'])

                    # If chosen you may have to mess around with this parameter to 
                        # widen/constrict the bounds that pop out
                    self['dynamic_multiplier'] = float(self['dynamic_multiplier'])

                    if 'absolute_bounds' in self:

                        self['absolute_bounds'] = [float(abs_bound) for abs_bound in self['absolute_bounds']]
                        self['bounds'] = self.construct_dynamic_bounds(
                            centre=self['default_value'], 
                            scaling=self['scaling'], 
                            dynamic_multiplier=self['dynamic_multiplier'], 
                            num_events=self['num_events'], 
                            absolute_bounds=self['absolute_bounds'])
                    else:
                        self['bounds'] = self.construct_dynamic_bounds(
                            centre=self['default_value'], 
                            scaling=self['scaling'], 
                            dynamic_multiplier=self['dynamic_multiplier'], 
                            num_events=self['num_events'],)
                        

            for idx, bound in enumerate(self['bounds']):
                self['bounds'][idx] = float(bound)
            if 'prob_model' in initial_data:
                self.prob_model = initial_data['prob_model']

            else:
                if self['scaling'] =='linear':
                        self.prob_model = uniform(
                            loc=self['bounds'][0],
                            scale=self['bounds'][1]-self['bounds'][0])
                if self['scaling'] =='log10':
                        self.prob_model = loguniform(
                            a=self['bounds'][0],
                            b=self['bounds'][1])

            # For sampling with 3 or less parameters we highly recommend making them discrete
                # and performing a scan over them rather than sampling as the performance is 
                # currently better than the stochastic/reweighting methods. Future updates
                # will try to improve the performance of the more expandable reweighting method
            if self['discrete']:

                if not('bins' in self) and (self['scaling'] == 'linear'):  

                    # Defaults to 10 per unit plus one so that if the bounds are 
                        # 0.0 --> 1.0 you get 
                        # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
                        # __not__
                        # 0.0, 0.111..., 0.222..., 0.333..., 0.444..., 0.555..., etc
                    self['bins'] = int(np.round(np.diff(self['bounds'])*10+1))

                elif not('bins' in self) and (self['scaling'] == 'log10'):  

                    # Defaults to 10 per unit plus one, mimicing the linear behaviour
                        # but in log10-space
                    self['bins'] = int(np.round(np.diff(np.log10(self['bounds']))*10+1))

                # The axis parameter is only required for discrete parameters
                if not('axis' in self):
                    if type(self['bins'])==int:
                        if self['scaling']=='linear':
                            self['axis'] = np.linspace(*self['bounds'], self['bins'])
                        else:
                            self['axis'] = np.logspace(np.log10(self['bounds'][0]), 
                                                    np.log10(self['bounds'][1]), 
                                                    self['bins'])
                            
                self['axis'] = np.asarray(self['axis'])
                            
                
                self['transform_scale'] = self['bins']
                if 'custom_parameter_transform' in self:
                    self.transform = self['custom_parameter_transform']
                else:
                    self.transform = self.discrete_parameter_transform
            else:

                # Future updates will allow one to set a custom function via a dynamic
                    # import
                if 'custom_parameter_transform' in self:
                    self.transform = self['custom_parameter_transform']
                else:
                    if self['scaling']=='linear':
                        self['transform_scale'] = float(np.diff(self['bounds']))
                        self.transform = self.continuous_parameter_transform_linear_scaling
                    else:
                        self['transform_scale'] = float(np.diff(np.log10(self['bounds'])))
                        self.transform = self.continuous_parameter_transform_log10_scaling

            # A parameter to keep track of which prior the parameter belongs to.
            if not('prior_id' in self):
                self['prior_id'] = np.nan

            # A parameter to keep track of which likelihood the parameter belongs to.
            if not('likelihood_id' in self):
                self['likelihood_id'] = np.nan

            # A parameter to keep track of what type of prior parameter this parameter is
                # atm (31/01/24) this is either 'spectral_parameters' or 'spatial_parameters'
            if not('parameter_type' in self):
                self['parameter_type'] = 'None'
                
        elif type(initial_data)==Parameter:
            # Handle duplicating from another Parameter instance
            super().__init__()
            for key, value in initial_data.items():
                self[key] = copy.deepcopy(value)
            # If there are any kwargs, update them as well.
            self.update(kwargs)

        else:
            super().__init__()
            if initial_data is not None:  # This ensures compatibility with empty or default init.
                raise TypeError("Initial data must be of type dict or Parameter")

            self.update(kwargs)


    @property
    def default(self):
        return self["default_value"]



    def construct_dynamic_bounds(self, centre, scaling, dynamic_multiplier, num_events, absolute_bounds=None):
        if scaling=='linear':
            bounds = [centre-dynamic_multiplier/np.sqrt(num_events), 
                            centre+dynamic_multiplier/np.sqrt(num_events)]
        else:
            bounds = [10**(np.log10(centre)-dynamic_multiplier/np.sqrt(num_events)), 
                            10**(np.log10(centre)+dynamic_multiplier/np.sqrt(num_events))]
            

        if not(absolute_bounds is None):
            if bounds[0]<absolute_bounds[0]:
                bounds[0] = absolute_bounds[0]

            if bounds[1]>absolute_bounds[1]:
                bounds[1] = absolute_bounds[1]

        return bounds
            
        

        


    # Below two methods allows one to access attributes like dictionary keys
    def __getattr__(self, item):
        """
        Allows attribute access like dictionary key access, providing a dynamic way 
        to interact with parameter properties.

        Args:
            item (str): The attribute (or dictionary key) to retrieve.

        Raises:
            AttributeError: If the attribute does not exist.

        Returns:
            The value associated with the key (or attribute name).
        """
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'Parameter' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        """
        Allows setting attributes like dictionary key-value pairs.

        Args:
            key (str): The attribute (or dictionary key) to set.
            value: The value to assign to the key.
        """
        self[key] = value
        super().__setattr__(key, value)


    # Method is only for discrete parameters
    @property
    def size(self):
        """
        Returns the size of the parameter's axis, applicable for discrete parameters.

        Returns:
            int: The size of the discrete parameter's axis.
        """
        return self.axis.size
    
    # Method is only for discrete parameters
    @property
    def shape(self):
        """
        Returns the shape of the parameter's axis, applicable for discrete parameters.

        Returns:
            tuple: The shape of the discrete parameter's axis.
        """
        return self.axis.shape


    def discrete_parameter_transform(self, u: float):
        """
        Transforms a uniform random variable into a discretely scaled parameter value.

        Args:
            u (float): A uniform random variable in the range [0, 1].

        Returns:
            float: The corresponding discrete parameter value.
        """

        scaled_value = u * self.transform_scale
        index = int(np.floor(scaled_value))
        # Ensure index is within bounds
        index = max(0, min(index, self.bins - 1))
        u = self.axis[index]

        return u
    
    def continuous_parameter_transform_linear_scaling(self, u: float):
        """
        Transforms a uniform random variable into a continuously scaled parameter value 
        using linear scaling.

        Args:
            u (float): A uniform random variable in the range [0, 1].

        Returns:
            float: The corresponding continuous parameter value under linear scaling.
        """

        u = u * self.transform_scale + self.bounds[0]

        return u

    
    def continuous_parameter_transform_log10_scaling(self, u: float):
        """
        Transforms a uniform random variable into a continuously scaled parameter value 
        using logarithmic (base 10) scaling.

        Args:
            u (float): A uniform random variable in the range [0, 1].

        Returns:
            float: The corresponding continuous parameter value under logarithmic scaling.
        """

        u = 10**(u * self.transform_scale + np.log10(self.bounds[0]))

        return u        
    

    

    def save(self, file_name: str):
        """
        Serializes the parameter object to an HDF5 file.

        Args:
            file_name (str): The name of the file to save the parameter data.
        """
        if not(file_name.endswith(".h5")):
            file_name = file_name + ".h5"

        with h5py.File(file_name, 'w-') as h5f:
            for key, value in self.items():
                if key!='transform' and key!='custom_parameter_transform':
                    # Convert arrays to a format that can be saved in h5py
                    if isinstance(value, (np.ndarray,list)):
                        h5f.create_dataset(key, data=value)
                    elif isinstance(value, (int, float, str, bool)):
                        h5f.attrs[key] = value
                    else:
                        # For all other types, we convert to string to ensure compatibility
                        h5f.attrs[key] = str(value)
                elif 'custom_parameter_transform' in self and hasattr(self['custom_parameter_transform'], '__call__'):
                    func_name = self['custom_parameter_transform'].__name__
                    h5f.attrs['custom_scaling_function_name'] = func_name
    
    
    def save_to_pickle(self, file_name: str):
        """
        Saves the parameter object to an pickle file.

        Args:
            file_name (str): The name of the file to save the parameter data.
        """

        if not(file_name.endswith('.pkl')):
                    file_name = file_name+'.pkl'

        pickle.dump(self, open(file_name,'wb'))



    @classmethod
    def load(cls, file_name: str):
        """
        Loads and initializes a Parameter object from an HDF5 file.

        Args:
            file_name (str): The name of the HDF5 file from which to load parameter data.

        Returns:
            Parameter: An initialized Parameter object with data loaded from the file.
        """
        with h5py.File(file_name, 'r') as h5f:
            data = {}
            # Load data stored as attributes
            for key, value in h5f.attrs.items():
                data[key] = value
            
            # Load data stored in datasets
            for key in h5f.keys():
                data[key] = np.array(h5f[key])

            print(data)
            return cls(data)
        
    @classmethod
    def load_from_pickle(cls, file_name: str):
        """
        Loads and initializes a Parameter object from an pickle file.

        Args:
            file_name (str): The name of the HDF5 file from which to load parameter data.

        Returns:
            Parameter: An initialized Parameter object with data loaded from the file.
        """
        if not(file_name.endswith(".pkl")):
            file_name = file_name + ".pkl"

        
        return  pickle.load(open(file_name,'rb'))
    

    def pdf(self, x):
        return self.prob_model.pdf(x)
    

    def logpdf(self, x):
        return self.prob_model.logpdf(x)
    
    def cdf(self, x):
        return self.prob_model.cdf(x)
    
    def logcdf(self, x):
        return self.prob_model.logcdf(x)
    
    @property
    def median(self, x):
        return self.prob_model.median()
    @property
    def mean(self, x):
        return self.prob_model.mean()
    @property
    def var(self):
        return self.prob_model.var()
    
    @property
    def std(self, x):
        return self.prob_model.std()
    
    def interval(self, confidence):
        return self.prob_model.interval(confidence)

    def ppf(self, q):
        return self.prob_model.ppf(q)
    
    
    
    
    