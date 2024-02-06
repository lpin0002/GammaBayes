import numpy as np, warnings, dynesty, logging
from gammabayes.utils import apply_direchlet_stick_breaking_direct, update_with_defaults
from gammabayes.hyper_inference.utils import _handle_parameter_specification
from gammabayes import ParameterSet, Parameter


class ScanOutput_ScanMixtureFracPosterior(object):

    def __init__(self, 
                 log_margresults, 
                 mixture_parameter_specifications: list | ParameterSet = None, 
                 log_nuisance_marg_regularisation = 0., 
                 prior_parameter_specifications = None # Argument is for consistent input to this class and "ScanOutput_StochasticMixtureFracPosterior"
                 ):
        
        self.log_nuisance_marg_results = log_margresults
        self.log_nuisance_marg_regularisation = log_nuisance_marg_regularisation
        self.mixture_parameter_specifications = self._handle_mixture_input(mixture_parameter_specifications)



    def _handle_mixture_input(self, 
                                 mixture_param_specifications: list | dict | ParameterSet = None, 
                                 log_margresults: list | tuple | np.ndarray = None):
        """
        Filters and validates the input for mixture parameters and log marginalisation results. 
        This method ensures that the provided mixture parameters and log marginalisation results 
        are consistent with the expected format and the current state of the object.

        Args:
            mixture_param_specifications (list | dict | ParameterSet, optional): The set of parameters for 
                                                                    the mixture model. Can be a list, 
                                                                    a dictionary, or a ParameterSet object. 
                                                                    Defaults to None.
            log_margresults (list | tuple | np.ndarray, optional): Log marginalisation results. 
                                                                Can be a list, tuple, or numpy ndarray. 
                                                                Defaults to None.

        Returns:
            tuple: A tuple containing the validated mixture_param_specifications and log_margresults.

        Raises:
            Exception: Raised if mixture_param_specifications is not specified or if the number of mixture 
                    axes does not match the number of components (minus 1) in the log priors.

        Notes:
            - If `mixture_param_specifications` is not provided, it defaults to `self.mixture_param_specifications`.
            - An exception is raised if no mixture axes are specified.
            - If `log_margresults` is not provided, it defaults to `self.log_margresults`.
            - The method checks the consistency between the number of mixture axes and the number 
            of prior components.
        """

        
        if mixture_param_specifications is None:
            mixture_param_specifications = self.mixture_param_specifications
            if self.mixture_param_specifications is None:
                raise Exception("Mixture axes not specified")
        else:
            self.mixture_param_specifications = ParameterSet(mixture_param_specifications)

        if not(log_margresults is None):
            if len(mixture_param_specifications)>len(log_margresults)-1:
                raise Exception(f""""There are more mixtures than would be implied by the log marginalisation results. Please check your inputs.
Number of mixture axes is {len(mixture_param_specifications)}
and number of prior components is {len(log_margresults)}.""")
            elif len(mixture_param_specifications)<len(log_margresults)-1:
                warnings.warn("""There are less mixtures than would be implied by the log marginalisatio results.
                              Assigning remaining mixtures as linearly uniform from 0 to 1 with 21 bins.""")
                for missing_mix in range(len(mixture_param_specifications), len(log_margresults)-1):
                    mixture_param_specifications.append(Parameter(
                        name=f'UnknownMix{missing_mix}',
                        bounds=[0., 1.],
                        discrete=True,
                        bins=21,
                    ))
            

        return mixture_param_specifications
    
    def create_mixture_comp(self, 
                            prior_idx: int, 
                            log_margresults_for_idx:np.ndarray,
                            mix_axes_mesh:list[np.ndarray],
                            final_output_shape:list,
                            prior_axes_indices: list[list],
                            hyper_idx: int):
        """
        Creates a single component (i.e. the component for __a__ prior) of the mixture model. 
        This method combines log marginalisation results for a given prior index with the 
        mixture axis information to form a component of the mixture model. 

        Args:
            prior_idx (int): The index of the prior for which the mixture component is being created.

            log_margresults_for_idx (np.ndarray): The log marginalisation results corresponding to the given prior index.

            mix_axes_mesh (list[np.ndarray]): A list of numpy arrays representing the meshgrid of mixture axes.

            final_output_shape (list): A list to store the final shape of the output mixture component.

            prior_axes_indices (list[list]): A list to keep track of the indices in each prior for the final output.

            hyper_idx (int): The current index in the hyperparameter space.

        Returns:
            tuple: A tuple containing the created mixture component (numpy array) and the updated hyperparameter index (int).

        Notes:
            - The method calculates the log mixture component using the Dirichlet stick breaking process.
            - It expands and combines the calculated mixture component with the log marginalisation results.
            - Updates `final_output_shape` and `prior_axes_indices` to reflect the new dimensions and indices after 
            combining the mixture component with the log marginalisation results.
            - The method returns the new mixture component and the updated hyperparameter index.
        """
        # Including 'event' and mixture axes for eventual __non__ expansion into
        single_prior_axes_indices_instance = list(range(1+len(mix_axes_mesh)))

        # First index of 'log_margresults_for_idx' should just be the number of events
        for length_of_axis in log_margresults_for_idx.shape[1:]:

            final_output_shape.append(length_of_axis)

            single_prior_axes_indices_instance.append(hyper_idx)

            # Keeping track of the indices in each prior for the final output
            hyper_idx+=1

        prior_axes_indices.append(single_prior_axes_indices_instance)


        mixcomp = np.expand_dims(np.log(
            apply_direchlet_stick_breaking_direct(mixtures_fractions=mix_axes_mesh, depth=prior_idx)), 
            axis=(*np.delete(np.arange(log_margresults_for_idx.ndim+len(mix_axes_mesh)), 
                                np.arange(len(mix_axes_mesh))+1),)) 


        mixcomp=mixcomp+np.expand_dims(log_margresults_for_idx, 
                                    axis=(*(np.arange(len(mix_axes_mesh))+1),)
                                )
        return mixcomp, hyper_idx
        
    
            
    def create_discrete_mixture_log_hyper_likelihood(self, 
                                                     batch_of_log_margresults: list | tuple | np.ndarray = None,
                                                     ) -> np.ndarray:

        batch_of_log_margresults = [log_margresult + self.log_nuisance_marg_regularisation for log_margresult in batch_of_log_margresults]
        

        mix_axes_mesh = np.meshgrid(*self.mixture_parameter_specifications.axes, indexing='ij')

        # To get around the fact that every component of the mixture can be a different shape
        final_output_shape = [len(batch_of_log_margresults), *np.arange(len(self.mixture_parameter_specifications)), ]

        # Axes for each of the priors to __not__ expand in
        prior_axes_indices = []

        # Counter for how many hyperparameter axes we have used
        hyper_idx = len(self.mixture_parameter_specifications)+1


        # Creating components of mixture for each prior
        mixture_array_comp_list = []
        for prior_idx, log_margresults_for_idx in enumerate(batch_of_log_margresults):

            mix_comp, hyper_idx  = self.create_mixture_comp( 
                            prior_idx = prior_idx, 
                            log_margresults_for_idx = log_margresults_for_idx,
                            mix_axes_mesh=mix_axes_mesh,
                            final_output_shape=final_output_shape,
                            prior_axes_indices=prior_axes_indices,
                            hyper_idx=hyper_idx)
            
            mixture_array_comp_list.append(mix_comp)
            
        # Making all the mixture components the same shape that is compatible for adding together (in logspace)
        for _prior_idx, mixture_array in enumerate(mixture_array_comp_list):
            axis = []
            for _axis_idx in range(len(final_output_shape)):
                if not(_axis_idx in prior_axes_indices[_prior_idx]):
                    axis.append(_axis_idx)
            axis = tuple(axis)

            mixture_array   = np.expand_dims(mixture_array, axis=axis)

            mixture_array_comp_list[_prior_idx] = mixture_array

        # Now to combine the results for all the data and to 
        combined_mixture = -np.inf
        for mixture_component in mixture_array_comp_list:
            mixture_component = mixture_component - self.log_nuisance_marg_regularisation
            combined_mixture = np.logaddexp(combined_mixture, mixture_component)


        # Multiplying the likelihoods of all the events together
        log_hyperparameter_likelihood = np.sum(combined_mixture, axis=0)

        # Undoing the regularisation for proper output
        log_hyperparameter_likelihood = log_hyperparameter_likelihood

        return log_hyperparameter_likelihood
    
    def initiate_exploration(self, 
                           num_in_batches: int) -> list:
        batched_log_margresults = []
        for batch_idx in range(0, len(self.log_nuisance_marg_results[0]), num_in_batches):
            batched_log_margresults.append(
                [single_prior_log_margresults[batch_idx:batch_idx+num_in_batches, ...] for single_prior_log_margresults in self.log_nuisance_marg_results])

        self.batched_log_nuisance_marg_results =  batched_log_margresults


    def run_exploration(self) -> np.ndarray:

        
        log_hyperparameter_likelihood_batches = [self.create_discrete_mixture_log_hyper_likelihood(
            batch_of_log_margresults=batch_of_log_margresults,
        ) for batch_of_log_margresults in self.batched_log_nuisance_marg_results]

        log_hyperparameter_likelihood = np.sum(log_hyperparameter_likelihood_batches, axis=0)

        return log_hyperparameter_likelihood