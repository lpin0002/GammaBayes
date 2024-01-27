from gammabayes.core import Parameter
import warnings, logging
import numpy as np

class ParameterSet(object):

    def __init__(self, parameter_specifications: dict = None):

        self.parameter_axes_by_type = {'spectral_parameters':{}, 'spatial_parameters':{}}
        self.dict_of_parameters_by_name = {}



        if (type(parameter_specifications) is dict):
            try:
                initial_type = type(list(parameter_specifications.values())[0])
            except IndexError as indexerr:
                print(f"An error occurred: {indexerr}")
                initial_type = None


            # Actually meant to be dict of parameter specifications by type->name->specifications?
            if (initial_type is dict):

                try:
                    secondary_type_is_dict = type(list(list(parameter_specifications.values())[0].values())[0]) == dict
                except IndexError as indexerr:
                    print(f"An error occurred: {indexerr}")
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

        elif (type(parameter_specifications) is tuple):
            unformatted_parameter_specifications = list(parameter_specifications)
            parameter_specifications = {unformatted_parameter_specifications[0]: unformatted_parameter_specifications[1]}
            
            if type(unformatted_parameter_specifications[1]) is dict:
                self.handle_plain_dict_input(parameter_specifications)

            elif type(unformatted_parameter_specifications[1]) is Parameter:
                self.handle_dict_of_param_input(parameter_specifications)

        elif (type(parameter_specifications)==list):

            # Actually meant to be list of parameter objects?
            if (type(parameter_specifications[0]) is Parameter):
                self.handle_list_param_input(parameter_specifications)

            elif (type(parameter_specifications[0]) is dict):
                self.handle_list_specification_input(parameter_specifications)

        elif isinstance(parameter_specifications, ParameterSet):
            self.append(parameter_specifications)            



    def handle_plain_dict_input(self, dict_input):
        for prior_name, single_prior_parameter_specifications in dict_input.items():
            for type_key, type_value in single_prior_parameter_specifications.items():
                for param_name, param_specification in type_value.items():
                    parameter = self.create_parameter(param_specification=param_specification, 
                                                      param_name=param_name, 
                                                      type_key=type_key, 
                                                      prior_name=prior_name)
                    self.append(parameter)


    def handle_name_dict_input(self, dict_input):
        for param_name, single_parameter_specifications in dict_input.items():

            parameter = self.create_parameter(param_specification=single_parameter_specifications, 
                                              param_name=param_name)
            self.append(parameter)

    def handle_dict_of_param_input(self, dict_input):
        for param_name, parameter_object in dict_input.items():
            self.append(parameter_object)


    def handle_list_param_input(self, list_input):
        for parameter_object in list_input:
            self.append(parameter_object)

    def handle_list_specification_input(self, list_input):
        for parameter_specification in list_input:
            parameter_object = Parameter(parameter_specification)
            self.append(parameter_object)



    def create_parameter(self, param_specification, param_name=None, type_key=None, prior_name=None):
        parameter = Parameter(param_specification)

        if type_key is not None:
            parameter['parameter_type'] = type_key

        if param_name is not None:
            parameter['name'] = param_name

        if prior_name is not None:
            parameter['prior_id'] = prior_name

        return parameter

    def append(self, parameter):
        if isinstance(parameter, Parameter):
            type_key = parameter['parameter_type']
            param_name = parameter['name']

            if type_key in self.parameter_axes_by_type:
                if 'axis' in parameter:
                    self.parameter_axes_by_type[type_key][param_name] = parameter['axis']
                else:
                    self.parameter_axes_by_type[type_key][param_name] = np.nan

            self.dict_of_parameters_by_name[param_name] = parameter

        elif isinstance(parameter, ParameterSet):
            parameters = parameter
            for param_name, parameter in parameters.dict_of_parameters_by_name.items():
                self.append(parameter)


    def __len__(self):
        return len(self.dict_of_parameters_by_name)



    def __add__(self, other):
        if not isinstance(other, ParameterSet):
            raise TypeError("Can only add another ParameterSet")

        combined = ParameterSet()
        for param_name, parameter in self.dict_of_parameters_by_name.items():
            combined.append(parameter)

        for param_name, parameter in other.dict_of_parameters_by_name.items():
            combined.append(parameter)  # This will overwrite if param_name already exists

        return combined
        

    @property
    def scan_format(self):
        return self.parameter_axes_by_type
    

    @property
    def stochastic_format(self):
        return self.dict_of_parameters_by_name
    

    @property
    def axes(self):
        return [param['axis'] for param in self.dict_of_parameters_by_name.values()]
    
    @property
    def defaults(self):
        try:
            return [param['default_value'] for param in self.dict_of_parameters_by_name.values()]
        except KeyError:
            logging.warn("One or more parameters have not been given a default value")
            return [np.nan]*len(self)
    
    def __iter__(self):
        return iter(self.dict_of_parameters_by_name)

