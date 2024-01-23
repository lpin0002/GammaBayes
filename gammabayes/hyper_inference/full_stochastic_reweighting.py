from scipy import special
from scipy.interpolate import interp1d
from tqdm import tqdm
import numpy as np
import functools, dynesty, warnings, os, sys, time
from matplotlib import pyplot as plt
from gammabayes.utils.event_axes import derive_edisp_bounds, derive_psf_bounds
from gammabayes.utils import update_with_defaults, iterate_logspace_integration, apply_direchlet_stick_breaking_direct
from gammabayes.samplers.sampler_utils import dynesty_restricted_proposal_marg_wrapper
from multiprocessing import Pool
from gammabayes import Parameter


class dynesty_stochastic_reweighting_class(object):

    def __init__(self, measured_events, irf_loglike, proposal_prior, target_priors, 
                 nuisance_axes, mixture_fraction_specifications=None,
                 logenergy_bound_percentile=90, logenergy_bound_sigmalevel=4, logenergy_bound=None, 
                 lonlat_bound_percentile=68, lonlat_bound_sigmalevel=4, lonlat_bound=None,
                 num_cores = 8,
                 logspace_integrator=iterate_logspace_integration,
                 parameter_specifications={}, reweight_batch_size=1000):
        self.measured_events                            = np.asarray(measured_events)
        self.irf_loglike                                = irf_loglike
        self.proposal_prior                             = proposal_prior
        self.target_priors                              = target_priors
        self.num_priors = len(self.target_priors)
        self.num_target_priors                          = len(target_priors)
        self.nuisance_axes                              = nuisance_axes
        self.mixture_fraction_specifications            = mixture_fraction_specifications
        self.logspace_integrator                        = logspace_integrator
        self.parameter_specifications                   = parameter_specifications
        self.reweight_batch_size                        = reweight_batch_size
        self.num_cores                                  = num_cores



        self.logenergy_bound = logenergy_bound
        self.lonlat_bound = lonlat_bound

        if self.logenergy_bound is None:
            _, self.logenergy_bound    = derive_edisp_bounds(irf_loglike=irf_loglike, percentile=logenergy_bound_percentile, sigmalevel=logenergy_bound_sigmalevel)            


        if self.lonlat_bound is None:
            self.lonlat_bound       = derive_psf_bounds(irf_loglike=irf_loglike, percentile=lonlat_bound_percentile, sigmalevel=lonlat_bound_sigmalevel)
            
        print(f"\nRestriction bounds: {self.logenergy_bound, self.lonlat_bound}\n\n")

    def run_proposal_dynesty(self, measured_event, NestedSampler_kwargs={'nlive':300}, run_nested_kwargs={'print_progress':True, 'dlogz':0.5, 'maxcall':10000}):
        
        dynesty_wrapper_instance = dynesty_restricted_proposal_marg_wrapper(measured_event=measured_event, irf_loglike=self.irf_loglike, proposal_prior=self.proposal_prior,
                                                    nuisance_axes=self.nuisance_axes,
                                                    logenergy_bound=self.logenergy_bound, lonlat_bound=self.lonlat_bound
                                                    )
    
        update_with_defaults(NestedSampler_kwargs, {'ndim': 3})

        # Set up the Nested Sampler with the multiprocessing pool
        sampler = dynesty.NestedSampler(dynesty_wrapper_instance.log_likelihood, dynesty_wrapper_instance.prior_transform, 
                                        **NestedSampler_kwargs)

        # Run Nested Sampling
        sampler.run_nested(**run_nested_kwargs)


        # Extract the results
        results = sampler.results

        # Computing the evidence
        log_evidence = results.logz
        
        return log_evidence[-1], results.samples
    
    def process_batch_proposal_dynesty(self, measured_events, NestedSampler_kwargs, run_nested_kwargs):
        log_batch_evidence_values = []
        batch_posterior_samples = []

        for measured_event in measured_events:
            single_event_results = self.run_proposal_dynesty(measured_event, 
                                                        NestedSampler_kwargs=NestedSampler_kwargs, 
                                                        run_nested_kwargs=run_nested_kwargs)
            log_batch_evidence_values.append(single_event_results[0])
            batch_posterior_samples.append(single_event_results[1])

        return log_batch_evidence_values, batch_posterior_samples
        

    
    def generate_proposal_posteriors(self, events_in_batch=100, num_cores=None, measured_events = None, 
                                     NestedSampler_kwargs={'nlive':100}, run_nested_kwargs={'print_progress':True, 'dlogz':0.5, 'maxcall':5000}):
        
        if num_cores is None:
            num_cores = self.num_cores

        if measured_events is None:
            measured_events = self.measured_events
        
        batched_event_measurements = []
        for batch_idx in range(0, len(measured_events), events_in_batch):
            batch_of_event_measurements = measured_events[batch_idx:batch_idx+events_in_batch, ...]
            batched_event_measurements.append(batch_of_event_measurements)


        partial_process_func = functools.partial(self.process_batch_proposal_dynesty, 
                                                 NestedSampler_kwargs=NestedSampler_kwargs, 
                                                 run_nested_kwargs=run_nested_kwargs)
        
        log_evidence_values = []
        posterior_samples = []

        with Pool(num_cores) as pool:
            for result in pool.imap(partial_process_func, tqdm(batched_event_measurements, 
                                                               total=len(batched_event_measurements), 
                                                               desc='Processing proposal batches')):
                log_evidence_values.extend(result[0])
                posterior_samples.extend(result[1])


        self.log_proposal_evidence_values   = log_evidence_values
        self.proposal_posterior_samples     = posterior_samples

        return log_evidence_values, posterior_samples
    

        
    def prior_transform(self, u):

        t1 = time.perf_counter()

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
    

    def init_reweighting_sampler(self, batch_size=None, mixture_fraction_specifications=None, parameter_specifications=None, **kwargs):

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

    
        self.sampler = dynesty.NestedSampler(loglikelihood=self.ln_likelihood, 
                                               prior_transform=self.prior_transform, 
                                               ndim=self.ndim, 
                                               **kwargs)
        
        return self.sampler
    
    def run_nested(self,*args, **kwargs):

        self.run_nested_output = self.sampler.run_nested(*args, **kwargs)

        return self.run_nested_output
