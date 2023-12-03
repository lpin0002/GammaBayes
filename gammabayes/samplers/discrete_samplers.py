import numpy as np
from gammabayes.utils.event_axes import energy_true_axis, longitudeaxistrue, latitudeaxistrue
import dynesty
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from scipy.special import logsumexp
from scipy.interpolate import interp1d
from functools import partial
from multiprocessing import Pool
import time
from .sampler_utils import construct_constrained_axes, default_proposal_prior_array, discrete_prior_transform
# Define the log likelihood function for your model

# sigma1 = 0.04
# sigma2 = 0.02
# sigma3 = 0.02

from dynesty import pool as dypool

class constrained_discrete_marginalisation(object):
    from gammabayes.likelihoods.irfs import single_loglikelihood


    def __init__(self, functional_likelihood=single_loglikelihood, prior=None,
                 axes=[energy_true_axis, longitudeaxistrue, latitudeaxistrue],
                 constrained_axes=None, 
                 axes_sigmas=[0.05,0.07,0.07], num_sigmas=8, sampler_instance=None,
                 parallelize = True, numcores=4):
        
        self.functional_likelihood      = functional_likelihood
        self.prior                      = prior
        self.axes                       = axes
        self.construct_constrained_axes = constrained_axes
        self.axes_sigmas                = axes_sigmas
        self.num_sigmas                 = num_sigmas
        self.sampler_instance           = sampler_instance
        self.parallelize                = parallelize
        self.numcores                   = numcores


    def construct_prior_transform_inputs(self, prior=None, constrained_axes=None,):
        logjacob = np.meshgrid(np.log(constrained_axes[0]), constrained_axes[1], constrained_axes[2], indexing='ij')[0]

        if prior is None:
            prior=self.prior
            if prior is None:
                log_prior_array = default_proposal_prior_array(constrained_axes)
            else:
                temp_log_prior_array = np.squeeze(prior.construct_prior_array(axes=constrained_axes,normalise=True))
                log_prior_array = temp_log_prior_array+logjacob
        else:
            log_prior_array = np.squeeze(prior.construct_prior_array(axes=constrained_axes,normalise=True))+logjacob

        flattened_logpriorarray = log_prior_array.flatten()
        logcdfarray = np.logaddexp.accumulate(flattened_logpriorarray)
        cdfarray = np.exp(logcdfarray-logcdfarray[-1])

        indices = np.arange(len(flattened_logpriorarray))
        inv_cdf_func = interp1d(x=cdfarray, y = indices, bounds_error=False, fill_value=(indices[0],indices[-1]), kind='nearest')

        return inv_cdf_func, log_prior_array
    





    def run(self, measured, ndim=3, nlive=200, functional_likelihood=None,prior=None, axes=None, maxcall=5000, dlogz=0.5,
            NestedSampler_kwargs={}, run_nested_kwargs={}):
        if axes is None:
            axes = self.axes
        if functional_likelihood is None:
            functional_likelihood = self.functional_likelihood
        if prior is None:
            prior = self.prior

        constrained_axes = construct_constrained_axes(measured=measured, axes=axes, num_sigmas=self.num_sigmas, axes_sigmas=self.axes_sigmas)
        inv_cdf_func, log_prior_array = self.construct_prior_transform_inputs(prior=prior, constrained_axes=constrained_axes,)
        # Create a NestedSampler or DynamicNestedSampler instance
        if not self.parallelize:
            sampler = NestedSampler(loglikelihood=functional_likelihood, 
                                    prior_transform=discrete_prior_transform, ndim=ndim, nlive=nlive, 
                                    ptform_kwargs={'axes':constrained_axes, 'inv_cdf_func':inv_cdf_func, 'log_prior_array':log_prior_array},
                                    logl_kwargs={'recon_energy':measured[0], 
                                                 'recon_lon':measured[1], 
                                                 'recon_lat':measured[2]}, 
                                    **NestedSampler_kwargs)
            # Run the sampler
            sampler.run_nested(dlogz=dlogz, maxcall=maxcall, **run_nested_kwargs)

            self.sampler_instance = sampler
        else:
            with dypool.Pool(self.numcores, loglike=functional_likelihood, prior_transform=discrete_prior_transform, 
                             ptform_kwargs={'inv_cdf_func':inv_cdf_func, 'log_prior_array':log_prior_array, 'axes':constrained_axes,},
                             logl_kwargs={'recon_energy':measured[0], 'recon_lon':measured[1], 'recon_lat':measured[2]}, ) as pool:
                sampler = NestedSampler(loglikelihood=pool.loglike, prior_transform=pool.prior_transform, 
                                        ndim=ndim, nlive=nlive, pool=pool, queue_size=self.numcores,
                                        **NestedSampler_kwargs)

                # Run the sampler
                sampler.run_nested(dlogz=dlogz, maxcall=maxcall, **run_nested_kwargs)

                self.sampler_instance = sampler


        return sampler


class bruteforce_prior_reweighting(object):
    def __init__(self, proposal_prior, target_prior, sampler_results):
        self.proposal_prior         = proposal_prior
        self.target_prior           = target_prior
        self.sampler_results        = sampler_results


        self.list_of_samples        = [result.samples_equal() for result in sampler_results]
        self.proposal_logz_results  = [result.logz[-1] for result in sampler_results]

        self.original_num_samples   = [len(samples) for samples in self.list_of_samples]
        max_num_samples             = np.max(self.original_num_samples)

        # We pad the samples with samples that evaluate to 0 so that vectorisation can easily be achieved
        padded_samples              = np.empty(shape=(len(sampler_results), max_num_samples, 3))

        for idx, samples in enumerate(self.list_of_samples):
            padded_samples[idx, ...] = np.append(np.asarray(samples), np.full(shape=(max_num_samples-len(samples),3), fill_value=[200., 0., 0.]), axis=0)

        self.padded_samples         = padded_samples

    def reweight_paramval(self, parameter_vals):

        matched_shape_param_vals = [self.padded_samples[:,:,2]*0+parameter_val for parameter_val in parameter_vals]


        matched_shape_param_vals_flattened = [mesh.flatten() for mesh in matched_shape_param_vals]

        target_log_prior_vals   = self.target_prior.logfunction(self.padded_samples[:,:,0].flatten() , self.padded_samples[:,:,1].flatten() , self.padded_samples[:,:,2].flatten() , *matched_shape_param_vals_flattened).reshape(self.padded_samples.shape[0],self.padded_samples.shape[1])
        
        proposal_log_prior_vals = self.proposal_prior(self.padded_samples)


        ratios = target_log_prior_vals-proposal_log_prior_vals-self.target_prior.normalisation(hyperparametervalues=parameter_vals)


        return np.sum(self.proposal_logz_results-np.log(self.original_num_samples)+logsumexp(ratios, axis=1))
    

    def reweight_with_param_axis(self, param_axis):
        return [self.reweight_paramval(parameter_val) for parameter_val in param_axis]
    


        
