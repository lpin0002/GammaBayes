from .parameter_set_class import ParameterSet
from .parameter_class import Parameter
from .core_utils import update_with_defaults
import numpy as np

class ParameterSetCollection:
    """A class for storing multiple ParameterSet instances and to handle mixture parameters
    and shared parameters between priors when doing analysis.

        Args:
            parameter_sets (list[ParameterSet]): _description_
            mixture_parameter_set (ParameterSet): _description_
            shared_parameters (dict[str, list[list[str], Parameter]]): _description_
            parameter_meta_data (dict, optional): _description_. Defaults to {}.
            collection_name (str, optional): _description_. Defaults to "".
    """

    def __init__(self, 
                 parameter_sets: list[ParameterSet],
                 mixture_parameter_set: ParameterSet,
                 shared_parameters: dict[str, list[list[str], Parameter]] = {},
                 parameter_meta_data: dict = {},
                 collection_name = "",
                 ):
        """_summary_

        Args:
            parameter_sets (list[ParameterSet]): _description_
            mixture_parameter_set (ParameterSet): _description_
            shared_parameters (dict[str, list[list[str], Parameter]]): _description_
            parameter_meta_data (dict, optional): _description_. Defaults to {}.
            collection_name (str, optional): _description_. Defaults to "".
        """
        self.collection_name = collection_name
        self.parameter_sets = []
        self.prior_ids_to_idx = {}
        for param_set_idx, parameter_set in enumerate(parameter_sets):
            if type(parameter_set) == ParameterSet:
                temp_formatted_parameter_set = parameter_set
            else:
                temp_formatted_parameter_set = ParameterSet(parameter_set)

            self.parameter_sets.append(temp_formatted_parameter_set)
            self.prior_ids_to_idx[temp_formatted_parameter_set.set_name] = param_set_idx


        self.shared_parameters = shared_parameters

        self.parameter_meta_data = parameter_meta_data

        self.mixture_parameter_set = mixture_parameter_set

        self.setup_discrete_prior_parameter_transform_intermediaries()


    def setup_discrete_prior_parameter_transform_intermediaries(self):
        self.prior_transform_list = []
        self.unique_parameter_list = []
        

        # This dictionary is so that once you have a value for a given index in the unit
            # cube, you need to go back to the log nuisance marginalisation matrices
            # and slice the relevant part of the matrix for the given parameter value
            # for the scanned parameters (i.e. the prior parameters are generally presumed
            # to be discrete for now)
        self.hyper_param_index_to_info_dict = {}
        self.prior_idx_to_param_idx_dict = {}
        hyper_param_idx = 0


        # First few indices of the unit cube are predesignated to be for the mixture weights
            # just for a consistent convention
        for mixture_param in self.mixture_parameter_set.values():

            self.hyper_param_index_to_info_dict[hyper_param_idx] = {'name': mixture_param['name'], 
                                                                    'prior_param_axes': [],
                                                                    'log_nuisance_marg_slice_indices':[],
                                                                    }

            if 'dependent' in mixture_param:
                self.hyper_param_index_to_info_dict[hyper_param_idx]['dependent'] = mixture_param['dependent']



            self.prior_transform_list.append([mixture_param.unitcube_transform, [hyper_param_idx]])
            self.unique_parameter_list.append([mixture_param, [hyper_param_idx]])
            hyper_param_idx+=1



        # Next few indices of the unit cube are predesignated for any shared parameters, again
            # just for a consistent convention
        for [prior_identifiers, shared_param] in self.shared_parameters.values():
            shared_param = Parameter(shared_param)

            prior_identifiers_indices = []
            for prior_identifier in prior_identifiers:
                if prior_identifier in self.prior_ids_to_idx:
                    prior_identifiers_indices.append(self.prior_ids_to_idx[prior_identifier])
        
            self.hyper_param_index_to_info_dict[hyper_param_idx] = {'name': shared_param['name'], 
                                                                    'prior_identifiers': prior_identifiers_indices, 
                                                                    'prior_param_axes': [],
                                                                    'log_nuisance_marg_slice_indices':[],
                                                                    }

            if 'dependent' in shared_param:
                self.hyper_param_index_to_info_dict[hyper_param_idx]['dependent'] = shared_param['dependent']


            self.prior_transform_list.append([shared_param.unitcube_transform, [hyper_param_idx]])
            self.unique_parameter_list.append([shared_param, [hyper_param_idx]])
            hyper_param_idx+=1



        # Remaining indices of the unit cube are allocated to the unique prior parameters in order of
            # the input of said priors to class then via input of spectral components then 
            # spatial/angular. The ordering of parameters within a parameter set for a given prior
            # is tracked via the 'hyper_param_idx' variable, which thanks to the ParameterSet class,
            # iterates along the parameters in the mentioned order.
            # I cannot think of a better way of doing this so feel free to send
            # me an email at Liam.Pinchbeck@monash.edu if you can think of something better
            
        for prior_idx, prior_param_set in enumerate(self.parameter_sets):
            self.prior_idx_to_param_idx_dict[prior_idx] = []
            prior_param_idx = 0
            for param_idx, param in enumerate(prior_param_set.values()):
                if not(param['discrete']):
                    raise ValueError(
f"""{param['name']} is not discrete. Prior parameters are presumed to be unique for this analysis class.
""")
                is_shared = False
                for temp_prior_param_idx, temp_parameter_info in self.hyper_param_index_to_info_dict.items():

                    # See if the parameter is shared
                    if param['name'] == temp_parameter_info['name']:
                                                                        # 
                        self.prior_idx_to_param_idx_dict[prior_idx].append([param_idx, prior_param_idx])
                        temp_parameter_info['prior_identifiers'].append(prior_idx)
                        temp_parameter_info['prior_param_axes'].append(param['axis'])
                        temp_parameter_info['log_nuisance_marg_slice_indices'].append(prior_param_idx)
                        is_shared = True

                    
                if not(is_shared):
                    self.prior_transform_list.append([param.unitcube_transform, [hyper_param_idx]])
                    self.unique_parameter_list.append([param, [hyper_param_idx]])

                    self.hyper_param_index_to_info_dict[hyper_param_idx] = {'name': param['name'], 
                                                                    'prior_identifiers': [prior_idx], 
                                                                    'prior_param_axes': [param['axis']],
                                                                    'log_nuisance_marg_slice_indices':[prior_param_idx],
                                                                    }
                    if 'dependent' in param:
                        self.hyper_param_index_to_info_dict[hyper_param_idx]['dependent'] = param['dependent']

                    self.prior_idx_to_param_idx_dict[prior_idx].append([param_idx, prior_param_idx])
                    hyper_param_idx+=1

                prior_param_idx+=1

            self.prior_idx_to_param_idx_dict[prior_idx] = np.asarray(self.prior_idx_to_param_idx_dict[prior_idx])


        remove_from_unique = set()
        dummy_parameter_transform_hyper_param_names = {}
        for param_idx, param_info in self.hyper_param_index_to_info_dict.items():
            if param_info['name'] in dummy_parameter_transform_hyper_param_names:
                self.prior_transform_list[param_idx][0] = self._dummy_prior_transform
                self.prior_transform_list[dummy_parameter_transform_hyper_param_names[param_info['name']]][1].append(param_idx)
                self.unique_parameter_list[dummy_parameter_transform_hyper_param_names[param_info['name']]][1].append(param_idx)
                remove_from_unique.add(param_idx)
            
            elif 'dependent' in param_info:
                for dependent_param_name in param_info['dependent']:
                    dummy_parameter_transform_hyper_param_names[dependent_param_name] = param_idx

        remove_from_unique = list(remove_from_unique)


        # Remove repeated parameters
        for index in sorted(remove_from_unique, reverse=True):
            del self.unique_parameter_list[index]






    def _dummy_prior_transform(self, u):
        return u



    def __repr__(self):
        string_val = "ParameterSetCollection Instance:\n"
        string_val+= f"Name:                {self.collection_name}\n"
        string_val+= f"Num Priors:          {len(self.parameter_sets)}\n"
        string_val+= f"Num Mixtures:        {len(self.mixture_parameter_set)}\n"
        string_val+= f"Num Shared:          {len(self.shared_parameters)}\n"
        string_val+= f"Num Unique Params:   {len(self.hyper_param_index_to_info_dict)}\n"
        return string_val


    def __len__(self):
        """
        Returns the number of unique parameters in the collection.

        Returns:
            int: The number of parameters.
        """
        return len(self.hyper_param_index_to_info_dict)



    def prior_transform(self, u):

        for prior_transform, prior_transform_indices in self.prior_transform_list:
            u[prior_transform_indices] = prior_transform(u[prior_transform_indices])
        return u
    
    def logpdf(self, input):

        output = np.zeros_like(input)

        for parameter, input_indices in self.unique_parameter_list:
            output[input_indices] = parameter.logpdf(input[input_indices])
        


        return np.sum(output)
    

    def pdf(self, input):
        return np.exp(self.logpdf(input))
    



