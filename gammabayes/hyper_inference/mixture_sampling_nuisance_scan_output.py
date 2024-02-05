import numpy as np, warnings, dynesty, logging
from gammabayes.utils import apply_direchlet_stick_breaking_direct, update_with_defaults
from gammabayes.hyper_inference.utils import _handle_parameter_specification
from gammabayes import ParameterSet


class ScanOutput_StochasticMixtureFracPosterior(object):

    def __init__(self, mixture_parameter_specifications, # e.g. mixture_bounds = [[mix1_min, mix1_max], [mix2_min, mix2_max],...]
                 prior_parameter_specifications,
                 log_margresults = []):
        
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
                print('hyper_idx: ', prior_idx, 'spectral_parameters', hyp_name, hyper_idx, idx_for_prior)
                index_to_hyper_parameter_info[hyper_idx] = [prior_idx, 'spectral_parameters', hyp_name, hyp_axis, idx_for_prior]
                idx_for_prior+1
                hyper_idx+=1

            for hyp_name, hyp_axis in prior_hyper_axes['spatial_parameters'].items():
                print('hyper_idx: ', prior_idx, 'spatial_parameters', hyp_name, hyper_idx, idx_for_prior)
                index_to_hyper_parameter_info[hyper_idx] = [prior_idx, 'spatial_parameters', hyp_name, hyp_axis, idx_for_prior]
                idx_for_prior+1
                hyper_idx+=1

        self.index_to_hyper_parameter_info = index_to_hyper_parameter_info

        self.ndim = self.num_mixes + len(self.index_to_hyper_parameter_info)




    def prior_transform(self, u):

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
    def initiate_sampler(self, **kwargs):

        self.sampler = dynesty.NestedSampler(loglikelihood=self.ln_likelihood, 
                                               prior_transform=self.prior_transform, 
                                               ndim=self.ndim, 
                                               **kwargs)
        
        return self.sampler
    
    def run_nested(self,*args, **kwargs):
        self.run_nested_output = self.sampler.run_nested(*args, **kwargs)
        return self.run_nested_output
        