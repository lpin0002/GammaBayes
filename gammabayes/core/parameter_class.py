import warnings, numpy as np
import copy, h5py, pickle
from scipy.stats import uniform, loguniform
from scipy import stats
from scipy.stats import rv_discrete
from scipy.stats import gamma


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

            self.update_distribution()
            

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

                if self.distribution is None:
                    self.distribution = rv_discrete(values=(self['axis'], np.ones_like(self['axis'])/len(self['axis'])))

                    # Making it so that if a "0" is given to the ppf it outputs the first discrete value
                    self.distribution.transform = self.distribution.ppf
                    self.distribution.ppf = self._adjusted_ppf
                    # pdf and pmf are essentially the same for the purposes of GammaBayes
                    self.distribution.pdf = self.distribution.pmf
                    self.distribution.logpdf = self.distribution.logpmf

                    self.rvs_sampling = False

            else:

                if not(self.distribution is None):
                    self.rvs_sampling = False
                    if hasattr(self.distribution, 'ppf'):
                        self.rvs_sampling = False
                    else:
                        self.rvs_sampling = True

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
            self.update_distribution()


        else:
            super().__init__()
            if initial_data is not None:  # This ensures compatibility with empty or default init.
                raise TypeError("Initial data must be of type dict or Parameter")



    def update_distribution(self):
        """
        Updates the parameter's distribution based on specified attributes.
        Supports custom distributions with optional keyword arguments.
        """
        self.custom_dist = False
        self.distribution = None
        
        if ('custom_dist_name' in self):
            self.custom_dist         = True

            dist_name = self['custom_dist_name']
            dist_kwargs = self['custom_dist_kwargs']

            self.distribution = getattr(stats, dist_name)(**dist_kwargs)

        elif 'custom_distribution' in self:
            
            self.custom_dist         = True

            self.distribution = self['custom_distribution']

            if type(self.distribution)==str:
                self['custom_dist_name'] = self.distribution

                del self.distribution
                del self['custom_distribution']

                self.update_distribution()

            elif 'custom_dist_kwargs' in self:
                dist_kwargs = self['custom_dist_kwargs']
                self.distribution = self.distribution(**dist_kwargs)




    @property
    def default(self):
        """
        Retrieves the default value of the parameter.

        Returns:
            float: The default value of the parameter.
        """
        return self["default_value"]



    def construct_dynamic_bounds(self, centre, scaling, dynamic_multiplier, num_events, absolute_bounds=None):
        """
        Constructs dynamic bounds for the parameter based on the number of events.

        Args:
            centre (float): Central value for the bounds.
            scaling (str): Scaling type ('linear' or 'log10').
            dynamic_multiplier (float): Multiplier for the dynamic adjustment.
            num_events (float): Number of events influencing the bounds.
            absolute_bounds (list, optional): Absolute bounds to constrain the dynamic bounds. Defaults to None.

        Returns:
            list: Computed dynamic bounds.
        """
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


    

    def pdf(self, x):
        """
        Computes the probability density function (PDF) at a given value (using 
        self.distribution.pdf, a scipy dist)

        Args:
            x (float or array-like): Value(s) at which to evaluate the PDF.

        Returns:
            float or array-like: PDF value(s).
        """
        return self.distribution.pdf(x)
    

    def logpdf(self, x):
        """
        Computes the log of the probability density function (logPDF) at a given value
        using `self.distribution.logpdf(x)`.

        Args:
            x (float or array-like): Value(s) at which to evaluate the logPDF.

        Returns:
            float or array-like: logPDF value(s).
        """
        return self.distribution.logpdf(x)
    
    def cdf(self, x):
        """
        Computes the cumulative distribution function (CDF) at a given value
        using `self.distribution.cdf(x)`.

        Args:
            x (float or array-like): Value(s) at which to evaluate the CDF.

        Returns:
            float or array-like: CDF value(s).
        """
        return self.distribution.cdf(x)
    
    def logcdf(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.distribution.logcdf(x)
    
    @property
    def median(self, x):
        """
        Computes the median of the distribution using 
        `self.distribution.median()`.

        Returns:
            float: Median value.
        """
        return self.distribution.median()
    
    @property
    def mean(self, x):
        """
        Computes the mean of the distribution using `self.distribution.mean()`.

        Returns:
            float: Mean value.
        """
        return self.distribution.mean()
    
    @property
    def var(self):
        """
        Computes the variance of the distribution using
        `self.distribution.var()`.

        Returns:
            float: Variance value.
        """
        return self.distribution.var()
    
    @property
    def std(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.distribution.std()
    
    def interval(self, confidence):
        """
        Computes the confidence interval for the distribution using
        `self.distribution.interval(confidence)`.

        Args:
            confidence (float): Confidence level.

        Returns:
            tuple: Lower and upper bounds of the confidence interval.
        """
        return self.distribution.interval(confidence)

    def _adjusted_ppf(self, q):
        """
        Adjusted percent point function (PPF) to return the smallest discrete value for q=0.

        Args:
            q (float or array-like): Quantile(s) at which to evaluate the PPF.

        Returns:
            float or array-like: Adjusted PPF value(s).
        """
        q = np.asarray(q)  
        # Making it so that if a 0 input is given then it returns the smallest discrete value
        result = np.where(q == 0, self.distribution.xk[0], self.distribution.transform(q))
        return result
    

    def ppf(self, q):
        """
        Computes the percent point function (PPF) at a given quantile.

        Args:
            q (float or array-like): Quantile(s) at which to evaluate the PPF.

        Returns:
            float or array-like: PPF value(s).
        """
        return self.distribution.ppf(q)
    

    def unitcube_transform(self, u):
        """
        Transforms a unit cube sample into the parameter space using the parameter's 
        inverse cumulative distribution function (ppf) or random variate sampling.

        Args:
            u (array-like): Sample from a unit cube.

        Returns:
            array-like: Transformed sample in the parameter space.
        """

        if self.rvs_sampling:

            return self.distribution.rvs()
        
        return self.ppf(u)
        
    


    
    
    
    
    