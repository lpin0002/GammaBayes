from .parameter_set_class import ParameterSet
from .parameter_class import Parameter
from gammabayes.utils import update_with_defaults
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
                 shared_parameters: dict[str, list[list[str], Parameter]],
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
        self.raw_parameter_sets = []
        for parameter_set in parameter_sets:
            self.raw_parameter_sets.append(ParameterSet(parameter_set))

        self.shared_parameters = shared_parameters


        self.parameter_meta_data = parameter_meta_data
        self.mixture_parameter_set = mixture_parameter_set

        self.setup_discrete_prior_parameter_transform_intermediaries()


    def setup_discrete_prior_parameter_transform_intermediaries(self):
        self.prior_transform_list = []

        # This dictionary is so that once you have a value for a given index in the unit
            # cube, you need to go back to the log nuisance marginalisation matrices
            # and slice the relevant part of the matrix for the given parameter value
            # for the scanned parameters (i.e. the prior parameters are generally presumed
            # to be discrete for now)
        self.prior_param_index_to_info_dict = {}
        hyper_param_idx = 0


        # First few indices of the unit cube are predesignated to be for the mixture weights
            # just for a consistent convention
        for mixture_param in self.mixture_parameter_set.values():
            self.prior_transform_list.append(mixture_param.transform)
            hyper_param_idx+=1


        # Next few indices of the unit cube are predesignated for any shared parameters, again
            # just for a consistent convention
        for [prior_identifiers, shared_param] in self.shared_parameters.values():
                                                    
        
            self.prior_param_index_to_info_dict[hyper_param_idx] = {'name': shared_param['name'], 
                                                                    'prior_identifiers': prior_identifiers, 
                                                                    'prior_param_axes': [],
                                                                    'log_nuisance_marg_slice_indices':[],
                                                                    }
            hyper_param_idx+=1

            self.prior_transform_list.append(shared_param.transform)


        # Remaining indices of the unit cube are allocated to the unique prior parameters in order of
            # the input of said priors to class then via input of spectral components then 
            # spatial/angular. The ordering of parameters within a parameter set for a given prior
            # is tracked via the 'prior_param_idx' variable, which thanks to the ParameterSet class,
            # iterates along the parameters in the mentioned order.
            # I cannot think of a better way of doing this so feel free to send
            # me an email at Liam.Pinchbeck@monash.edu if you can think of something better
            
        for prior_idx, prior_param_set in enumerate(self.raw_parameter_sets):
            prior_param_idx = 0
            for param in prior_param_set.values():
                if not(param['discrete']):
                    raise ValueError(
f"""{param['name']} is not discrete. Prior parameters are presumed to be unique for this analysis class.
""")
                is_shared = False
                for param_info in self.prior_param_index_to_info_dict.values():

                    # See if the parameter is shared
                    if (param['name'] == param_info['name']) and (prior_idx in param_info['prior_identifiers']):
                        is_shared = True
                        param_info['log_nuisance_marg_slice_indices'].append(prior_param_idx)
                        param_info['prior_param_axes'].append(param['axis'])

                    
                if not(is_shared):
                    self.prior_transform_list.append(param.transform)



    def __repr__(self):
        string_val = "ParameterSetCollection Instance:\n"
        string_val+= f"Name:                {self.collection_name}\n"
        string_val+= f"Num Priors:          {len(self.raw_parameter_sets)}\n"
        string_val+= f"Num Mixtures:        {len(self.mixture_parameter_set)}\n"
        string_val+= f"Num Shared:          {len(self.shared_parameters)}\n"
        string_val+= f"Num Unique Params:   {len(self.prior_param_index_to_info_dict)}\n"
        return string_val


    def __len__(self):
        """
        Returns the number of unique parameters in the collection.

        Returns:
            int: The number of parameters.
        """
        return len(self.prior_param_index_to_info_dict)



    def prior_transform(self, u):
        u = np.column_stack([f(u[:, i]) for i, f in enumerate(self.prior_transform_list)])
        return u

