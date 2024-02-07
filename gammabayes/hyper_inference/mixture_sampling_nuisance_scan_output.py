import numpy as np, warnings, dynesty, logging
from gammabayes.utils import apply_direchlet_stick_breaking_direct, update_with_defaults
from gammabayes.hyper_inference.utils import _handle_parameter_specification
from gammabayes import ParameterSet


class ScanOutput_StochasticMixtureFracPosterior(object):
    """
    A class designed to handle stochastic exploration of posterior distributions for mixture fractions using the Dynesty sampler.

    Attributes:
        prior_parameter_specifications (list | dict | ParameterSet): Specifications for prior parameters.
        
        mixture_parameter_specifications (ParameterSet): Specifications for mixture parameters.
        
        log_margresults (np.array): Logarithm of marginal results for the parameters.
        
        log_nuisance_marg_regularisation (float): Regularisation parameter for nuisance marginalisation.
        
        num_priors (int): The number of prior distributions provided.
        
        num_events (int): The number of events in the log marginal results.
        
        mixture_bounds (list): Bounds for the mixture parameters.
        
        num_mixes (int): The number of mixture components.
        
        hyperparameter_axes (list): Axes for hyperparameters, derived from prior parameter specifications.
        
        index_to_hyper_parameter_info (dict): Mapping of hyperparameter indices to their specifications.
        
        ndim (int): The number of dimensions for the sampler, including mixture components and hyperparameters.
        
        sampler (dynesty.NestedSampler): The Dynesty sampler object used for exploration.
    """
    def __init__(self, 
                 log_margresults,
                 mixture_parameter_specifications:list | dict | ParameterSet,
                 log_nuisance_marg_regularisation: float = 0., # Argument for consistency between classes
                 prior_parameter_specifications: list | dict | ParameterSet = None):
        """
        Initializes the ScanOutput_StochasticMixtureFracPosterior class with necessary parameters and configurations.

        Args:
            log_margresults (np.array): Logarithm of marginal results for the parameters.
            
            mixture_parameter_specifications (list | dict | ParameterSet): Specifications for mixture parameters.
            
            log_nuisance_marg_regularisation (float, optional): Regularisation parameter for nuisance marginalisation.
            
            prior_parameter_specifications (list | dict | ParameterSet, optional): Specifications for prior parameters.
        """
        self.prior_parameter_specifications = _handle_parameter_specification(
            prior_parameter_specifications,
            _no_required_num=True
        )
        self.hyperparameter_axes    = [param_set.axes_by_type for param_set in self.prior_parameter_specifications]


        self.mixture_parameter_specifications = ParameterSet(mixture_parameter_specifications)
        self.mixture_bounds         = self.mixture_parameter_specifications.bounds


        self.log_margresults = log_margresults
        self.num_priors     = len(self.log_margresults)
        self.num_events     = log_margresults[0].shape[0]


        if not(len(self.mixture_bounds) +1 != len(log_margresults)):
            warnings.warn("""Number of mixtures +1 does not match number of 
priors indicated in log_margresults. Assigning min=0 and max=1 for remaining mixtures.""")
            
            assert len(log_margresults[0])>len(self.mixture_bounds) +1

            for missing_mix_idx in range(len(log_margresults)-(len(self.mixture_bounds) +1)):
                self.mixture_bounds.append([0., 1.])

        
        self.num_mixes              = len(self.mixture_bounds)


        # Counter for hyperparameter axes, mostly for 'index_to_hyper_parameter_info'
        hyper_idx = 0

        # Creating components of mixture for each prior
        self.num_hyper_axes         = 0
        index_to_hyper_parameter_info = {}

        for prior_idx, prior_hyper_axes in enumerate(self.hyperparameter_axes):
            idx_for_prior = 0

            update_with_defaults(prior_hyper_axes, {'spectral_parameters': {}, 'spatial_parameters':{}})

            self.num_hyper_axes+=len(prior_hyper_axes['spectral_parameters'])
            self.num_hyper_axes+=len(prior_hyper_axes['spatial_parameters'])

            for hyp_name, hyp_axis in prior_hyper_axes['spectral_parameters'].items():
                # print('hyper_idx: ', prior_idx, 'spectral_parameters', hyp_name, hyper_idx, idx_for_prior)
                index_to_hyper_parameter_info[hyper_idx] = [prior_idx, 'spectral_parameters', hyp_name, hyp_axis, idx_for_prior]
                idx_for_prior+1
                hyper_idx+=1

            for hyp_name, hyp_axis in prior_hyper_axes['spatial_parameters'].items():
                # print('hyper_idx: ', prior_idx, 'spatial_parameters', hyp_name, hyper_idx, idx_for_prior)
                index_to_hyper_parameter_info[hyper_idx] = [prior_idx, 'spatial_parameters', hyp_name, hyp_axis, idx_for_prior]
                idx_for_prior+1
                hyper_idx+=1

        self.index_to_hyper_parameter_info = index_to_hyper_parameter_info

        self.ndim = self.num_mixes + len(self.index_to_hyper_parameter_info)




    def prior_transform(self, u):
        """
        Transforms uniform samples `u` from the unit cube to the parameter space defined by mixture and hyperparameter axes.

        Args:
            u (np.array): An array of uniform samples to be transformed.

        Returns:
            np.array: Transformed samples in the parameter space.
        """

        for _mix_idx in range(self.num_mixes):
            u[_mix_idx] = self.mixture_bounds[_mix_idx][0]+u[_mix_idx]*(self.mixture_bounds[_mix_idx][1]- self.mixture_bounds[_mix_idx][0])

        for _hyper_idx in range(self.num_hyper_axes):
            hyper_axis = self.index_to_hyper_parameter_info[_hyper_idx][3]
            # Scale the value in u to the range of indices
            scaled_value = u[self.num_mixes + _hyper_idx] * len(hyper_axis)
            index = int(np.floor(scaled_value))
            # Ensure index is within bounds
            index = max(0, min(index, len(hyper_axis) - 1))
            u[self.num_mixes + _hyper_idx] = hyper_axis[index]
        return u
    
    def ln_likelihood(self, inputs):
        """
        Calculates the logarithm of the likelihood for a given set of input parameters.

        Args:
            inputs (np.array): Array of input parameters, including mixture weights and hyperparameter values.

        Returns:
            float: The logarithm of the likelihood for the given inputs.
        """
        
        mixture_weights = inputs[:self.num_mixes]
        hyper_values    = inputs[self.num_mixes:]

        slices_for_priors = [[] for idx in range(self.num_priors)]

        for hyper_idx in range(self.num_hyper_axes):
                slice_idx = np.where(self.index_to_hyper_parameter_info[hyper_idx][3]==hyper_values[hyper_idx])[0][0]
                slices_for_priors[self.index_to_hyper_parameter_info[hyper_idx][0]].append(slice_idx)

        ln_like = -np.inf
        for prior_idx in range(self.num_priors):
            ln_component = np.log(apply_direchlet_stick_breaking_direct(mixture_weights, depth=prior_idx))

            ln_marg_results_for_prior = self.log_margresults[prior_idx]
            try:
                ln_component += ln_marg_results_for_prior[:, *slices_for_priors[prior_idx]]
            except:
                ln_component += ln_marg_results_for_prior

            ln_like = np.logaddexp(ln_like, ln_component)

        return np.sum(ln_like)


    # Note: When allocating live points it's lower than the norm due to the discrete nature of the prior parameters
    def initiate_exploration(self, **kwargs):

        """
        Initiates the exploration process by setting up the Dynesty sampler with the class's likelihood and prior transform functions.

        Args:
            **kwargs: Keyword arguments to be passed to the Dynesty sampler.

        Returns:
            dynesty.NestedSampler: The configured Dynesty sampler object.
        """

        self.sampler = dynesty.NestedSampler(loglikelihood=self.ln_likelihood, 
                                               prior_transform=self.prior_transform, 
                                               ndim=self.ndim, 
                                               **kwargs)
        
        return self.sampler
    
    def run_exploration(self,*args, **kwargs):
        """
        Runs the nested sampling exploration process using the initialized Dynesty sampler.

        Args:
            *args: Positional arguments to be passed to the `run_nested` method of Dynesty sampler.
            **kwargs: Keyword arguments to be passed to the `run_nested` method of Dynesty sampler.

        Returns:
            dict: A dictionary containing the results of the nested sampling run.
        """
        self.run_nested_output = self.sampler.run_nested(*args, **kwargs)
        return self.run_nested_output
    
        