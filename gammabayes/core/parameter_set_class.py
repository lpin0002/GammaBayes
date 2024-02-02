from gammabayes.core import Parameter
import warnings, logging
import numpy as np
import h5py

class ParameterSet(object):
    """
    A collection of Parameter objects, organized for efficient manipulation, retrieval,
    and storage. Supports various input formats for initializing the set, including
    dictionaries, lists, and other ParameterSet instances.
    """

    def __init__(self, parameter_specifications: dict | list | tuple = None):
        """
        Initializes a ParameterSet with the given parameter specifications.

        Args:
            parameter_specifications (dict | list | ParameterSet | tuple, optional): A dictionary, list, tuple, or another
                ParameterSet instance containing the specifications for initializing the parameter set.
                This can include parameter names, types, prior names, and their specifications or instances.
        """

        self.axes_by_type = {'spectral_parameters':{}, 'spatial_parameters':{}}
        self.dict_of_parameters_by_name = {}



        if (type(parameter_specifications) == dict):
            try:
                initial_type = type(list(parameter_specifications.values())[0])
            except IndexError as indexerr:
                logging.warning(f"An error occurred: {indexerr}")
                initial_type = None


            # Actually meant to be dict of parameter specifications by type->name->specifications?
            if (initial_type is dict):

                try:
                    secondary_type_is_dict = type(list(list(parameter_specifications.values())[0].values())[0]) == dict
                except IndexError as indexerr:
                    logging.warning(f"An error occurred: {indexerr}")
                    secondary_type_is_dict = False 


                if secondary_type_is_dict:
                    self.handle_plain_dict_input(parameter_specifications)

                else:
                    self.handle_name_dict_input(parameter_specifications)
            # Actually meant to be dict of parameter specifications by name->specifications?
            elif (initial_type is Parameter):
                self.handle_dict_of_param_input(parameter_specifications)

            

            else:
                warnings.warn("Something went wrong when tring to access dictionary values.")

        elif (type(parameter_specifications) == tuple):
            unformatted_parameter_specifications = list(parameter_specifications)
            parameter_specifications = {unformatted_parameter_specifications[0]: unformatted_parameter_specifications[1]}
            
            if type(unformatted_parameter_specifications[1]) == dict:
                self.handle_plain_dict_input(parameter_specifications)

            elif type(unformatted_parameter_specifications[1]) == Parameter:
                self.handle_dict_of_param_input(parameter_specifications)

        elif (type(parameter_specifications)==list):

            # Actually meant to be list of parameter objects?
            if (type(parameter_specifications[0]) is Parameter):
                self.handle_list_param_input(parameter_specifications)

            elif (type(parameter_specifications[0]) is dict):
                self.handle_list_specification_input(parameter_specifications)

        elif isinstance(parameter_specifications, ParameterSet):
            self.append(parameter_specifications)            


    # This method presumes that you have given the specifications in the form of 
            # nested dictionaries of the form
            # prior_name > parameter type > parameter name > parameter specifications
    def handle_plain_dict_input(self, dict_input: dict[dict[dict[dict]]]):
        """
        Handles input in the form of a nested dictionary specifying parameter types, names,
        and specifications, structured by prior name.

        Args:
            dict_input (dict): The input dictionary containing parameter specifications.
        """

        
        for prior_name, single_prior_parameter_specifications in dict_input.items():
            for type_key, type_value in single_prior_parameter_specifications.items():
                for param_name, param_specification in type_value.items():
                    parameter = self.create_parameter(param_specification=param_specification, 
                                                      param_name=param_name, 
                                                      type_key=type_key, 
                                                      prior_name=prior_name)
                    self.append(parameter)

    # This method presumes that you have given the specifications in the form of 
            # nested dictionaries of the form
            # parameter name > parameter specifications
    def handle_name_dict_input(self, dict_input: dict[dict]):
        """
        Handles input where specifications are given in the form of a dictionary keyed
        by parameter name.

        Args:
            dict_input (dict): The input dictionary containing parameter specifications.
        """
        for param_name, single_parameter_specifications in dict_input.items():

            parameter = self.create_parameter(param_specification=single_parameter_specifications, 
                                              param_name=param_name)
            self.append(parameter)


    # This method presumes that you have given the specifications in the form of 
            # nested dictionaries of the form
            # parameter name > parameter class instances
    def handle_dict_of_param_input(self, dict_input: dict[Parameter]):
        """
        Processes input given as a dictionary of Parameter instances keyed by parameter name.

        Args:
            dict_input (dict): The input dictionary of Parameter instances.
        """
        for param_name, parameter_object in dict_input.items():
            self.append(parameter_object)

    # This method presumes that you have given the specifications in the form of 
            # a list of parameter objects
    def handle_list_param_input(self, list_input : list[Parameter]):
        """
        Accepts a list of Parameter objects and appends them to the set.

        Args:
            list_input (list): The list of Parameter objects.
        """
        for parameter_object in list_input:
            self.append(parameter_object)


    # This method presumes that you have given the specifications in the form of 
            # a list of dictionaries containing the specifications for the parameters
    def handle_list_specification_input(self, list_input: list[Parameter]):
        """
        Handles input given as a list of dictionaries, each containing specifications
        for a parameter.

        Args:
            list_input (list): The list of dictionaries with parameter specifications.
        """
        for parameter_specification in list_input:
            parameter_object = Parameter(parameter_specification)
            self.append(parameter_object)


    # A kind of filter class for creating parameters for use within the Set
    def create_parameter(self, 
                         param_specification: dict, 
                         param_name: str = None, 
                         type_key: str = None, 
                         prior_name: str = None):
        """
        Creates a Parameter object from given specifications, optionally setting its
        type, name, and associated prior name.

        Args:
            param_specification (dict): Specifications for the parameter.
            
            param_name (str, optional): Name of the parameter.
            
            type_key (str, optional): Type of the parameter (e.g., 'spectral_parameters').
            
            prior_name (str, optional): Name of the prior associated with the parameter.

        Returns:
            Parameter: The created Parameter object.
        """
        parameter = Parameter(param_specification)

        if type_key is not None:
            parameter['parameter_type'] = type_key

        if param_name is not None:
            parameter['name'] = param_name

        if prior_name is not None:
            parameter['prior_id'] = prior_name

        return parameter
    


    def append(self, parameter: Parameter):
        """
        Appends a Parameter object or another ParameterSet to this set.

        Args:
            parameter (Parameter or ParameterSet): The parameter or parameter set to append.
        """
        if isinstance(parameter, Parameter):
            type_key = parameter['parameter_type']

            param_name = parameter['name']

            if type_key in self.axes_by_type:
                if 'axis' in parameter:
                    self.axes_by_type[type_key][param_name] = parameter['axis']
                else:
                    self.axes_by_type[type_key][param_name] = np.nan

            self.dict_of_parameters_by_name[param_name] = parameter

        elif isinstance(parameter, ParameterSet):
            parameters = parameter
            for param_name, parameter in parameters.dict_of_parameters_by_name.items():
                self.append(parameter)


    def __len__(self):
        """
        Returns the number of parameters in the set.

        Returns:
            int: The number of parameters.
        """
        return len(self.dict_of_parameters_by_name)


    # Allows the behaviour of set1 + set2 = new combined set
    def __add__(self, other):
        """
        Allows the addition of two ParameterSets, combining their parameters into a new set.

        Args:
            other (ParameterSet): The other ParameterSet to add.

        Returns:
            ParameterSet: A new ParameterSet instance containing parameters from both sets.

        Raises:
            TypeError: If 'other' is not an instance of ParameterSet.
        """
        if not isinstance(other, ParameterSet):
            raise TypeError("Can only add another ParameterSet")

        combined = ParameterSet()
        for param_name, parameter in self.dict_of_parameters_by_name.items():
            combined.append(parameter)

        for param_name, parameter in other.dict_of_parameters_by_name.items():
            combined.append(parameter)  # This will overwrite if param_name already exists

        return combined
    

    # Below three methods are toreplicate straight dictionary behaviour
    def items(self):
        """
        Returns a view of the parameter set's items (parameter name, Parameter object pairs).

        Returns:
            ItemsView: A view of the parameter set's items.
        """
        return self.dict_of_parameters_by_name.items()
    
    def values(self):
        """
        Returns a view of the parameters in the set.

        Returns:
            ValuesView: A view of the Parameter objects in the set.
        """
        return self.dict_of_parameters_by_name.values()
    
    def keys(self):
        """
        Returns a view of the parameter names in the set.

        Returns:
            KeysView: A view of the parameter names.
        """
        return self.dict_of_parameters_by_name.keys()
        

    # Outputs the dictionary of the form used for analysis classes that explore 
        # parameter spaces via scan
    @property
    def scan_format(self):
        """
        Returns a representation of the parameter set formatted for scanning analysis classes.

        Returns:
            dict: A dictionary representation of the parameter set suitable for scanning.
        """
        return self.axes_by_type
    
    # Outputs the dictionary of the form used for analysis classes that explore 
        # parameter spaces via stochastic samplers
    @property
    def stochastic_format(self):
        """
        Returns a representation of the parameter set formatted for stochastic sampling analysis classes.

        Returns:
            dict: A dictionary representation of the parameter set suitable for stochastic sampling.
        """
        return self.dict_of_parameters_by_name
    
    # Method specifically when all parameters are discrete and have a defined 'axis' method
    @property
    def axes(self):
        """
        Returns the axes for all parameters in the set, applicable for discrete parameters.

        Returns:
            list: A list of axes for the parameters.
        """
        return [param['axis'] for param in self.dict_of_parameters_by_name.values()]
    
    # Ex
    @property
    def defaults(self):
        """
        Retrieves the default values for all parameters in the set.

        Returns:
            list: A list of default values for the parameters.
        """
        default_values = []
        for parameter in self.dict_of_parameters_by_name.values():

            try:
                default_values.append(parameter['default_value'])
            except KeyError:
                logging.warn("""One or more parameters have not been given a 
default value. Place nan in position of default""")
                default_values.append(np.nan)
        return default_values
    
    # When some sort of iterable behaviour is expected of the class it
        # defaults to using the behaviour of the dict_of_parameters_by_name dict
    def __iter__(self):
        """
        Allows iteration over the ParameterSet, yielding parameter names.

        This method facilitates direct iteration over the parameter set, providing an easy way
        to loop through all parameter names contained within it.

        Returns:
            iterator: An iterator over the names of parameters in the set.
        """
        return iter(self.dict_of_parameters_by_name)



    def pack(self, h5f):
        """
        Packs the ParameterSet data into an HDF5 format.

        Returns:
        h5py.File: An HDF5 file object containing the packed data.
        """
        if h5f is None:
            h5f = h5py.File(None, 'w')  # 'None' creates an in-memory HDF5 file object
            
        for param_name, parameter in self.dict_of_parameters_by_name.items():
            group = h5f.create_group(param_name)
            for key, value in parameter.items():
                
                if key != 'transform' and key != 'custom_parameter_transform':
                    # Convert arrays to a format that can be saved in h5py
                    if isinstance(value, (np.ndarray, list)):
                        group.create_dataset(key, data=value)
                    elif isinstance(value, (int, float, str, bool)):
                        group.attrs[key] = value
                    else:
                        group.attrs[key] = str(value)
                elif 'custom_parameter_transform' in parameter and hasattr(parameter['custom_parameter_transform'], '__call__'):
                    func_name = parameter['custom_parameter_transform'].__name__
                    group.attrs['custom_scaling_function_name'] = func_name
        return h5f
    
    def save(self, file_name):
        """
        Saves the ParameterSet data to an HDF5 file.

        Args:
        file_name (str): The name of the file to save the data to.
        """
        h5f = self.pack()
        with open(file_name, 'wb') as file:  # Open the file in binary write mode
            file.write(h5f.file.get_file_image())  # Write the in-memory HDF5 file to disk


    @classmethod
    def load(cls, h5f, file_name: str = None):
        """
        Loads the ParameterSet data from an HDF5 file and returns a ParameterSet instance.

        Args:
            file_name (str): The name of the file to load the data from.

        Returns:
            ParameterSet: An instance of the ParameterSet class with the loaded data.
        """
        have_to_close = False

        
        parameter_list  = []
        if h5f is None:
            have_to_close = True
            h5f = h5py.File(file_name, 'r') 
            
        for param_name in h5f.keys():
            group = h5f[param_name]
            param_data = {key: group.attrs[key] for key in group.attrs.keys()}

            for key in group.keys():
                param_data[key] = np.array(group[key])
            
            # Reconstruct the Parameter instance
            if param_data != {}:
                parameter = Parameter(param_data)

                parameter_list.append(parameter)

        if have_to_close:
            h5f.close()
        if parameter_list != []:
            return ParameterSet(parameter_list)
        else:
            return ParameterSet()
    