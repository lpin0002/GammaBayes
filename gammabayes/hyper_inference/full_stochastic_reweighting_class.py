from .scan_reweighting_class import DynestyScanReweighting
import dynesty, time, numpy as np
from gammabayes.utils import update_with_defaults, iterate_logspace_integration, apply_direchlet_stick_breaking_direct
from scipy import special
from gammabayes import Parameter, ParameterSet

class DynestyStochasticReweighting(DynestyScanReweighting):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # To ensure a mix of methods aren't used unless I create equivalent methods
    def reweight_single_param_combination(self, *args, **kwargs):
        raise NotImplementedError
    
    def reweight_single_prior(self, *args, **kwargs):
        raise NotImplementedError
    
    def scan_reweight(self, *args, **kwargs):
        raise NotImplementedError
    
    def select_scan_output_posterior_exploration_class(self, *args, **kwargs):
        raise NotImplementedError
    

    def init_posterior_exploration(self, *args, **kwargs):
        raise NotImplementedError
    
    def run_posterior_exploration(self, *args, **kwargs):
        raise NotImplementedError
    


    def prior_transform(self, u):

        for _mix_idx in range(self.num_mixes):
            u[_mix_idx] = self.mixture_parameters[_mix_idx].transform(u[_mix_idx])
        for _hyper_idx in range(self.num_hyper_axes):
            u[self.num_mixes + _hyper_idx] = self.parameters[_hyper_idx].transform(u[self.num_mixes + _hyper_idx])

        return u
    

    def evaluate_prior_on_batch(self, batch_proposal_posterior_samples, 
                                       prior, 
                                       prior_spectral_params_vals,
                                       prior_spatial_params_vals):
        
        prior_values = prior(
                        batch_proposal_posterior_samples[:,0],
                        batch_proposal_posterior_samples[:,1],
                        batch_proposal_posterior_samples[:,2],
                        spectral_parameters  = {spec_key:batch_proposal_posterior_samples[:,0]*0.+spec_val for spec_key, spec_val in prior_spectral_params_vals.items()}, 
                        spatial_parameters   = {spat_key:batch_proposal_posterior_samples[:,0]*0.+spat_val for spat_key, spat_val in prior_spatial_params_vals.items()})

        return prior_values
    

    def ln_likelihood(self, inputs):
        mixture_weights = inputs[:self.num_mixes]
        hyper_values    = inputs[self.num_mixes:]

        mixture_values_array = np.log([apply_direchlet_stick_breaking_direct(mixture_weights, depth=prior_id) for prior_id in range(self.num_target_priors)])


        prior_params_dicts    = [{'spectral_parameters':{}, 'spatial_parameters':{}} for target_prior_idx in range(self.num_target_priors)]

        for hyper_parameter, hyper_parameter_value in zip(self.parameters, hyper_values):
            prior_params_dicts[hyper_parameter.prior_id][hyper_parameter.parameter_type][hyper_parameter.name] = hyper_parameter_value
        
        target_prior_norms  = np.asarray([target_prior.normalisation(**prior_params_dict) for target_prior, prior_params_dict in zip(self.target_priors, prior_params_dicts)])
        target_prior_norms[np.where(np.isinf(target_prior_norms))] = 0


        ln_likelihood_values = []
        for log_evidence, samples in zip(self.log_proposal_evidence_values, self.proposal_posterior_samples):

    
            num_samples = len(samples[:,0])
            target_prior_values     = np.empty(shape=(num_samples,self.num_priors,))


            proposal_prior_values = \
                self.evaluate_prior_on_batch(batch_proposal_posterior_samples=samples, 
                                    prior=self.proposal_prior, 
                                    prior_spectral_params_vals={},
                                    prior_spatial_params_vals={})
            

            
            for prior_id, target_prior in enumerate(self.target_priors):
                
                
                target_prior_values[:, prior_id] = \
                    self.evaluate_prior_on_batch(batch_proposal_posterior_samples=samples, 
                                    prior=target_prior, 
                                    prior_spectral_params_vals=prior_params_dicts[prior_id]['spectral_parameters'],
                                    prior_spatial_params_vals=prior_params_dicts[prior_id]['spatial_parameters'],)
                    
            added_target_prior_values = special.logsumexp(target_prior_values+mixture_values_array[None, :]-target_prior_norms, axis=1)
        
            ln_likelihood_value = log_evidence - np.log(num_samples) + special.logsumexp(added_target_prior_values-proposal_prior_values+self.proposal_prior_ln_norm)
            
            ln_likelihood_values.append(ln_likelihood_value)

        ln_like = np.sum(ln_likelihood_values)
            
        return ln_like
    

    def init_reweighting_sampler(self, batch_size=None, mixture_fraction_specifications=None, parameter_specifications=None,sampling_class=dynesty.NestedSampler, **kwargs):

        if batch_size is None:
            batch_size = self.reweight_batch_size


        if not(mixture_fraction_specifications is None):
            self.mixture_fraction_specifications = mixture_fraction_specifications
            
        self.num_mixes = len(self.mixture_fraction_specifications)

        self.mixture_parameters = []

        for mix_name, mixture_parameter_specifications in self.mixture_fraction_specifications.items():
            mixture_parameter = Parameter(mixture_parameter_specifications)
            if not('name' in mixture_parameter):
                mixture_parameter['name'] = mix_name
                
            self.mixture_parameters.append(mixture_parameter)


        if not(parameter_specifications is None):
            self.parameter_specifications = parameter_specifications
        
        self.parameters = []
        for prior_id, prior_parameter_specifications in enumerate(self.parameter_specifications.values()):
            for parameter_name, parameter_specification in prior_parameter_specifications.items():
                print(parameter_name, parameter_specification)
                parameter               = Parameter(parameter_specification)
                parameter['name']       = parameter_name
                parameter['prior_id']   = prior_id
                self.parameters.append(parameter)

        self.num_hyper_axes = len(self.parameter_specifications)

        self.ndim = self.num_hyper_axes + self.num_mixes

        self.num_priors = len(self.target_priors)

        self.proposal_prior_ln_norm = self.proposal_prior.normalisation()

    
        self.sampler = sampling_class(loglikelihood=self.ln_likelihood, 
                                               prior_transform=self.prior_transform, 
                                               ndim=self.ndim, 
                                               **kwargs)
        
        return self.sampler
    
    def run_nested(self,*args, **kwargs):

        self.run_nested_output = self.sampler.run_nested(*args, **kwargs)

        return self.run_nested_output
    
    @property
    def posterior_exploration_results(self):
        return self.sampler.results
