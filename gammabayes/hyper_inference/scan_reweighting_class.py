from scipy import special
from scipy.interpolate import interp1d
from tqdm import tqdm
import numpy as np
import functools, dynesty, warnings, os, sys, time
from matplotlib import pyplot as plt
from gammabayes.utils.event_axes import derive_edisp_bounds, derive_psf_bounds
from gammabayes.utils import (
    update_with_defaults, 
    iterate_logspace_integration, 
    bound_axis, 
    apply_direchlet_stick_breaking_direct)

import logging
from gammabayes.priors import DiscreteLogPrior
from gammabayes.likelihoods import DiscreteLogLikelihood
from gammabayes import EventData, Parameter, ParameterSet
from gammabayes.hyper_inference.utils import _handle_parameter_specification, _handle_nuisance_axes
from gammabayes.hyper_inference.mixture_scan_nuisance_scan_output import ScanOutput_ScanMixtureFracPosterior
from gammabayes.hyper_inference.mixture_sampling_nuisance_scan_output import ScanOutput_StochasticMixtureFracPosterior





# Small class to generate proposal posterior samples for a single event
class DynestyProposalMarg(object):

    def __init__(self, 
                 measured_event: list | np.ndarray, 
                 log_proposal_prior: DiscreteLogPrior, 
                 log_likelihood: DiscreteLogLikelihood, 
                 nuisance_axes: list[np.ndarray] | tuple[np.ndarray], 
                 bounds: list[list[(str, float)]]):
        """
        Initializes an instance for generating proposal posterior samples for a single event,
        utilizing a nested sampling approach via the Dynesty sampler.

        Parameters:
            measured_event (list | np.ndarray): The observed event data, specified as a list or a numpy array.
            log_proposal_prior (DiscreteLogPrior): An instance of DiscreteLogPrior, representing the logarithm of the proposal prior.
            log_likelihood (DiscreteLogLikelihood): An instance of DiscreteLogLikelihood, representing the logarithm of the likelihood function.
            nuisance_axes (list[np.ndarray] | tuple[np.ndarray]): A list or tuple of numpy arrays, each representing a nuisance parameter axis.
            bounds (list[list[(str, float)]]): A nested list where each inner list contains tuples specifying the bounds for nuisance parameters. Each tuple contains a string indicating the bound type and a float indicating the bound value.
        """

        
        self.measured_event = measured_event

        restricted_axes_info = [
            bound_axis(
                nuisance_axis, 
                bound_type, 
                bound_radii, 
                estimated_val) for nuisance_axis, [bound_type, bound_radii], estimated_val in zip(nuisance_axes, bounds, measured_event)]
                        

        self.restricted_axes = [axis_info[0] for axis_info in restricted_axes_info]

        nuisance_mesh = np.meshgrid(*measured_event, *self.restricted_axes, indexing='ij')
        
        self.log_prior_matrix = np.squeeze(log_proposal_prior(*nuisance_mesh[3:]))
        
        self.logpdf_shape = self.log_prior_matrix.shape
        flattened_logpdf = self.log_prior_matrix.flatten()
        flattened_logcdf = np.logaddexp.accumulate(flattened_logpdf)
        flattened_logcdf = flattened_logcdf

        # Pseudo as isn't done with actual integration over the relevant axes
        self.pseudo_log_norm = flattened_logcdf[-1]

        if np.isfinite(self.pseudo_log_norm):
            flattened_logcdf = flattened_logcdf - self.pseudo_log_norm

        flattened_cdf = np.exp(flattened_logcdf)

        self.interpolation_func = interp1d(y=np.arange(flattened_cdf.size), x=flattened_cdf, 
                                    fill_value=(0, flattened_cdf.size-1), bounds_error=False)
        


        self.log_likelihood = functools.partial(
            log_likelihood.irf_loglikelihood.dynesty_single_loglikelihood,
            recon_energy=measured_event[0], recon_lon=measured_event[1], recon_lat=measured_event[2]
        )

        self.bounds = bounds

        logging.info(f"Bounds: {self.bounds}")


    def prior_transform(self, u: float):
        """
        Transforms a sample from the unit cube to the parameter space based on the prior distribution.

        Parameters:
            u (float): A sample from the unit cube.

        Returns:
            list: A list of parameter values transformed according to the prior distribution.
        """
        flat_idx = self.interpolation_func(u[0])
        flat_idx = int(np.round(flat_idx))

        restricted_axis_indices = np.unravel_index(shape=self.logpdf_shape, indices=flat_idx)

        axis_values = [restricted_axis[restricted_axis_index] for restricted_axis, restricted_axis_index in zip(self.restricted_axes, restricted_axis_indices)]
        
        return axis_values
    
    def likelihood(self, x: float):
        """
        Calculates the log-likelihood for a given set of parameters.

        Parameters:
            x (float): A parameter vector for which the log-likelihood is to be calculated.

        Returns:
            float: The log-likelihood of the given parameters.
        """

        log_like_val = self.log_likelihood(x)

        return log_like_val





class DynestyScanReweighting(object):

    def __init__(self, 
                 measured_events: EventData, 
                 log_likelihood: DiscreteLogLikelihood, 
                 log_proposal_prior: DiscreteLogPrior, 
                 log_target_priors: list[DiscreteLogPrior], 
                 nuisance_axes: list[np.ndarray] = None, 
                 mixture_param_specifications: dict | ParameterSet = None,
                 marginalisation_bounds: list[(str, float)] = None,
                 bounding_percentiles: list[float] = [90, 90],
                 bounding_sigmas: list[float] = [4,4],
                 logspace_integrator: callable = iterate_logspace_integration,
                 prior_parameter_specifications: dict | list[ParameterSet] = {}, 
                 reweight_batch_size: int = 1000,
                 no_priors_on_init: bool = False,
                 log_prior_normalisations:np.ndarray = None,
                 mixture_fraction_exploration_type='scan',
                 ):
        """
        Initializes an object to reweight nested sampling results based on a set of target priors.

        Parameters:
            measured_events (EventData): The event data to be analyzed.
            
            log_likelihood (DiscreteLogLikelihood): An instance representing the log-likelihood function.
            
            log_proposal_prior (DiscreteLogPrior): An instance representing the log of the proposal prior.
            
            log_target_priors (list[DiscreteLogPrior]): A list of instances representing the log of the target priors.
            
            nuisance_axes (list[np.ndarray], optional): Nuisance parameter axes (true energy, longitude and latitude).
            
            mixture_param_specifications (dict | ParameterSet, optional): Specifications for mixture model parameters.
            
            marginalisation_bounds (list[(str, float)], optional): Bounds for nuisance parameter marginalization.
            
            bounding_percentiles (list[float], optional): Percentiles used to calculate bounding for the nuisance 
            parameter marginalisation if 'marginalisation_bounds' is not given.
            
            bounding_sigmas (list[float], optional): Sigma levels used to calculate bounding for the nuisance 
            parameter marginalisation if 'marginalisation_bounds' is not given.
            
            logspace_integrator (callable, optional): A callable for log-space integration (multi dimensions).
            
            prior_parameter_specifications (dict | list[ParameterSet], optional): Parameter sets for the priors.
            
            reweight_batch_size (int, optional): Batch size for reweighting operation.
            
            no_priors_on_init (bool, optional): If True, no errors are raised if priors are not supplied on 
            initialization. 
            
            log_prior_normalisations (np.ndarray, optional): Normalisation constants for the log priors.
        """

        self.measured_events        = measured_events

        self.log_likelihood = log_likelihood
        self.log_proposal_prior = log_proposal_prior
        self.log_target_priors = log_target_priors
        self.no_priors_on_init = no_priors_on_init

        self.nuisance_axes = _handle_nuisance_axes(nuisance_axes, 
                                                   log_likelihood=self.log_likelihood,
                                                   log_prior=self.log_target_priors[0])
        self.mixture_param_specifications = mixture_param_specifications
        self.logspace_integrator = logspace_integrator

        if not self.no_priors_on_init:
            self.prior_parameter_specifications = _handle_parameter_specification(
                    parameter_specifications=prior_parameter_specifications,
                    num_required_sets=len(self.log_target_priors),
                    _no_required_num=self.no_priors_on_init)
        else:
            self.prior_parameter_specifications = _handle_parameter_specification(
                    prior_parameter_sets=prior_parameter_specifications,
                    _no_required_num=self.no_priors_on_init)        
            
        self._num_parameter_specifications = len(self.prior_parameter_specifications)

        self.reweight_batch_size = reweight_batch_size
        self.bounds = marginalisation_bounds
        self.log_prior_normalisations = log_prior_normalisations
        self.mixture_fraction_exploration_type = mixture_fraction_exploration_type


        if self.bounds is None:
            _, logeval_bound               = derive_edisp_bounds(irf_loglike=self.log_likelihood, percentile=bounding_percentiles[0], sigmalevel=bounding_sigmas[0])
            lonlact_bound                   = derive_psf_bounds(irf_loglike=self.log_likelihood, percentile=bounding_percentiles[1], sigmalevel=bounding_sigmas[1], 
                                                                axis_buffer=1, parameter_buffer = np.squeeze(np.ptp(self.dependent_axes[1]))/2)
            self.bounds = [['log10', logeval_bound], ['linear', lonlact_bound], ['linear', lonlact_bound]]

        logging.info(f"Bounds: {self.bounds}")


    
            


    # Function to get proposal posterior samples for a single event
    def run_proposal_dynesty(self, 
                             measured_event: list | np.ndarray, 
                             NestedSampler_kwargs: dict = {'nlive':200}, 
                             run_nested_kwargs: dict = {'dlogz':0.7, 
                                                        'maxcall':5000}):
        """
        Runs the Dynesty nested sampling algorithm for a single event using the proposal prior.

        Parameters:
            measured_event (list | np.ndarray): The event data.
            
            NestedSampler_kwargs (dict, optional): Keyword arguments for the NestedSampler.
            
            run_nested_kwargs (dict, optional): Keyword arguments for the run_nested method.

        Returns:
            object: The result of the nested sampling algorithm.
        """

        single_event_dynesty_wrapper = DynestyProposalMarg(
            measured_event=measured_event, 
            log_likelihood=self.log_likelihood, 
            log_proposal_prior=self.log_proposal_prior,
            nuisance_axes=self.nuisance_axes,
            bounds = self.bounds, 
        )
    
        update_with_defaults(NestedSampler_kwargs, {'ndim': 3})

        # Set up the Nested Sampler with the multiprocessing pool
        sampler = dynesty.NestedSampler(single_event_dynesty_wrapper.log_likelihood, 
                                        single_event_dynesty_wrapper.prior_transform, 
                                        **NestedSampler_kwargs)

        # Run Nested Sampling
        sampler.run_nested(**run_nested_kwargs)

        # Extract the results
        results = sampler.results


        return results
    

    # A function to allow one to run the proposal sampling on multiple events
    def batch_process_proposal_sampling(self,
                                        measured_events: EventData, 
                                        NestedSampler_kwargs: dict = {'nlive':300}, 
                                        run_nested_kwargs: dict = {'dlogz':0.7, 
                                                                   'maxcall':5000,
                                                                   'print_progress':False},
                                        ) -> list[(float, list)]:
        """
        Processes multiple events in batches using the Dynesty nested sampling algorithm.

        Parameters:
            measured_events (EventData): A collection of event data.
            
            NestedSampler_kwargs (dict, optional): Keyword arguments for the NestedSampler.
            
            run_nested_kwargs (dict, optional): Keyword arguments for the run_nested method.

        Returns:
            list[tuple]: A list of tuples containing log evidence and samples for each event.
        """
        likelihood_results  = []
        proposal_samples    = []

        for event_datum in measured_events:
            sampling_results = self.run_proposal_dynesty(
                             measured_event = event_datum, 
                             NestedSampler_kwargs=NestedSampler_kwargs, 
                             run_nested_kwargs=run_nested_kwargs)
            likelihood_results.append(sampling_results.logz[-1])
            proposal_samples.append(sampling_results.samples)


        num_samples = [len(proposal_sample) for proposal_sample in proposal_samples]

        logging.debug(f"Min, Median and Max log proposal evidence: {min(likelihood_results)}, {np.median(likelihood_results)}, {max(likelihood_results)}")
        logging.debug(f"Min, Median and Max for number of samples to explore proposal posterior: {min(num_samples)}, {np.median(num_samples)}, {max(num_samples)}")
            
        return likelihood_results, proposal_samples

        

    # Need to vectorise the use of parameters as well
    def reweight_single_param_combination(self, 
                                   log_evidence_values: list[float] | np.ndarray[float], 
                                   nuisance_parameter_samples: list[list[float]] | list[np.ndarray[float]], 
                                   proposal_prior_func: DiscreteLogPrior,
                                   target_prior_func: DiscreteLogPrior, 
                                   target_prior_spectral_params: dict[(str, np.ndarray)], 
                                   target_prior_spatial_params: dict[(str, np.ndarray)], 
                                   target_prior_ln_norm: float, 
                                   proposal_prior_ln_norm: float, 
                                   batch_size: int = None):
        """
        Reweights log evidence values for a single parameter combination by comparing the proposal and target prior 
        probabilities across a batch of nuisance parameter samples.

        Parameters:
            log_evidence_values (list[float] | np.ndarray[float]): Log evidence values obtained from nested sampling.
            
            nuisance_parameter_samples (list[list[float]] | list[np.ndarray[float]]): Nuisance parameter samples.
            
            proposal_prior_func (DiscreteLogPrior): The proposal prior.
            
            target_prior_func (DiscreteLogPrior): The target prior.
            
            target_prior_spectral_params (dict[(str, np.ndarray)]): Spectral parameters for the target prior.
            
            target_prior_spatial_params (dict[(str, np.ndarray)]): Spatial parameters for the target prior.
            
            target_prior_ln_norm (float): Log normalization constant for the target prior.
            
            proposal_prior_ln_norm (float): Log normalization constant for the proposal prior.
            
            batch_size (int, optional): The batch size for processing. Defaults to `self.reweight_batch_size` if None.

        Returns:
            list[float]: A list of reweighted log likelihood values.

        Note:
            - It is presumed that the proposal prior has no hyperparameters.
        """
        if batch_size is None:
            batch_size = self.reweight_batch_size
        
        ln_likelihood_values = []

        for log_evidence, samples in zip(log_evidence_values, nuisance_parameter_samples):
            num_samples = len(samples)
            proposal_prior_values   = np.empty(shape=(num_samples,))
            target_prior_values     = np.empty(shape=(num_samples,))

            for batch_idx in range(0, num_samples, batch_size):
                batch_slice = slice(batch_idx, batch_idx+batch_size)
                proposal_prior_values[batch_slice] = proposal_prior_func(*samples[batch_slice, :].T)

                target_spectral_parameter_mesh = {spec_key:samples[batch_slice,0]*0.+spec_val \
                                                  for spec_key, spec_val \
                                                    in target_prior_spectral_params.items()}
                target_spatial_parameter_mesh = {spat_key:samples[batch_slice,0]*0.+spat_val \
                                                 for spat_key, spat_val \
                                                    in target_prior_spatial_params.items()}
                
                target_prior_values[batch_slice] = target_prior_func(
                    *samples[batch_slice, :].T,
                    spectral_parameters  = target_spectral_parameter_mesh, 
                    spatial_parameters   = target_spatial_parameter_mesh,)
            
            # Uses the trick that the average of a function over a posterior is approximately
                    # equal to the sum of the values of the function on posterior samples
            ln_likelihood_value = log_evidence-np.log(num_samples) \
                + special.logsumexp(
                    target_prior_values-target_prior_ln_norm-proposal_prior_values+proposal_prior_ln_norm)
            ln_likelihood_values.append(ln_likelihood_value)


        return ln_likelihood_values
    
    def reweight_single_prior(self, 
                              prior_parameter_set, 
                              log_target_prior, 
                              nested_sampling_results_samples,
                              log_evidence_values,
                              proposal_prior_ln_norm):
        """
        Reweights nested sampling results for a single prior by applying the target prior probability to 
        the nested sampling results samples.

        Parameters:
            prior_parameter_set (ParameterSet): The set of parameters for the target prior.
            
            log_target_prior (DiscreteLogPrior): The target prior.
            
            nested_sampling_results_samples (list | np.ndarray): Nested sampling results samples.
            
            log_evidence_values (list | np.ndarray): Log evidence values obtained from nested sampling
            using the proposal prior.
            
            proposal_prior_ln_norm (float): Log normalization constant for the proposal prior.

        Returns:
            np.ndarray: An array of reweighted log marginal results for the single prior.
        """
        
        prior_parameter_axis_dict = prior_parameter_set.axes_by_type

        self.log_prior_normalisations[log_target_prior.name] = log_target_prior.normalisation(
            **prior_parameter_axis_dict)
        

        update_with_defaults(prior_parameter_axis_dict, 
                                {'spectral_parameters':{}, 
                                'spatial_parameters':{}})
        hyper_param_mesh = np.meshgrid(
            *prior_parameter_axis_dict['spectral_parameters'].values(),
            *prior_parameter_axis_dict['spatial_parameters'].values(),
            indexing='ij')
        
        num_spec_parameters = len(prior_parameter_axis_dict['spectral_parameters'].keys())


        hyper_param_dict_mesh = {}
        hyper_param_dict_mesh['spectral_parameters'] = {
            spec_key:hyper_param_mesh[spec_idx] for spec_idx, spec_key in enumerate(
                prior_parameter_axis_dict['spectral_parameters'].keys())}
    
        hyper_param_dict_mesh['spatial_parameters'] = {
            spat_key:hyper_param_mesh[spat_idx+num_spec_parameters] \
                for spat_idx, spat_key \
                    in enumerate(prior_parameter_axis_dict['spatial_parameters'].keys())}
        
        # Note: 0 ain't special, I'm just using it to access _a_ mesh
        try:
            example_mesh = hyper_param_mesh[0]
            num_hyper_indices = hyper_param_mesh[0].size
            example_mesh_shape = example_mesh.shape

        except (IndexError, ValueError) as indxerr:
            logging.debug(indxerr)
            num_hyper_indices = 1
            example_mesh_shape = (1,)



        target_prior_log_marg_results = np.empty(
            shape=(len(nested_sampling_results_samples), *example_mesh_shape))


        for hyper_val_idx in tqdm(range(num_hyper_indices), total=num_hyper_indices, leave=False):

            
            hyper_axis_indices = np.unravel_index(hyper_val_idx, example_mesh_shape)

            if num_hyper_indices>1:
                target_prior_ln_norm = self.log_prior_normalisations[log_target_prior.name][*hyper_axis_indices]
            else:
                target_prior_ln_norm = self.log_prior_normalisations[log_target_prior.name]
            #
            if not(np.isfinite(target_prior_ln_norm)):
                target_prior_ln_norm = 0

            single_target_prior_hyper_val_log_marg_results = np.squeeze(
                np.asarray(
                    self.reweight_single_param_combination(
                        log_evidence_values=log_evidence_values, 
                        nuisance_parameter_samples      = nested_sampling_results_samples,
                        target_prior_func               = log_target_prior, 
                        proposal_prior_func             = self.log_proposal_prior,

                        target_prior_spectral_params    = {
                            key:axis[hyper_axis_indices[idx]] \
                                for idx, (key, axis) in enumerate(prior_parameter_axis_dict['spectral_parameters'].items())},
                        target_prior_spatial_params     = {
                            key:axis[hyper_axis_indices[idx+num_spec_parameters]] \
                                for idx, (key, axis) in enumerate(prior_parameter_axis_dict['spatial_parameters'].items())},
                        
                        target_prior_ln_norm            = target_prior_ln_norm, 
                        proposal_prior_ln_norm          = proposal_prior_ln_norm
                        )
                        )
                        )

            
            target_prior_log_marg_results[:, *hyper_axis_indices] = single_target_prior_hyper_val_log_marg_results

        return target_prior_log_marg_results
    
    def scan_reweight(self, 
                      log_evidence_values: list | np.ndarray, 
                      nested_sampling_results_samples: list | np.ndarray, 
                      log_target_priors: list[DiscreteLogPrior] = None,
                      prior_parameter_specifications: dict | list[ParameterSet] = None):
        """
        Scans and reweights log evidence values against multiple target priors by applying a reweighting procedure 
        for each target prior and parameter set combination.

        Parameters:
            log_evidence_values (list | np.ndarray): Log evidence values obtained from nested sampling.
            
            nested_sampling_results_samples (list | np.ndarray): The resultant samples from nested sampling using 
            proposal prior.
            
            log_target_priors (list[DiscreteLogPrior], optional): A list of target priors. If None, uses 
            `self.log_target_priors`.
            
            prior_parameter_specifications (dict | list[ParameterSet], optional): Parameter sets for each prior. 
            If None, uses `self.prior_parameter_specifications`.

        Returns:
            list[np.ndarray]: A list of reweighted log marginal results for each target prior and parameter set combination.
        """
        if log_target_priors is None:
            log_target_priors = self.log_target_priors


        if prior_parameter_specifications is None:
            prior_parameter_specifications = self.prior_parameter_specifications
        else:
            prior_parameter_specifications = _handle_parameter_specification(
                parameter_specifications = prior_parameter_specifications,
                num_required_sets=len(log_target_priors),)


        normalisation_mesh = np.meshgrid(*self.nuisance_axes, indexing='ij')
        flattened_meshes = [mesh.flatten() for mesh in normalisation_mesh]


        proposal_prior_matrix= np.squeeze(
                self.log_proposal_prior(
                    *flattened_meshes, 
                    ).reshape(normalisation_mesh[0].shape))
            
        proposal_prior_ln_norm = iterate_logspace_integration(proposal_prior_matrix, axes=self.nuisance_axes)

        if self.log_prior_normalisations is None:
            self.log_prior_normalisations = {}


        nuisance_log_marg_results = []


        for log_target_prior, prior_parameter_set in tqdm(
            zip(log_target_priors, prior_parameter_specifications), 
            total=len(log_target_priors)):
            

            target_prior_log_marg_results = self.reweight_single_prior(prior_parameter_set, 
                                        log_target_prior, 
                                        nested_sampling_results_samples,
                                        log_evidence_values,
                                        proposal_prior_ln_norm)

            nuisance_log_marg_results.append(target_prior_log_marg_results)


        self.nuisance_log_marg_results = nuisance_log_marg_results


        log_marg_mins = np.asarray([np.nanmin(nuisance_log_marg_result[nuisance_log_marg_result != -np.inf]) for nuisance_log_marg_result in self.nuisance_log_marg_results])
        log_marg_maxs = np.asarray([np.nanmax(nuisance_log_marg_result[nuisance_log_marg_result != -np.inf]) for nuisance_log_marg_result in self.nuisance_log_marg_results])

        # Adaptively set the regularisation based on the range of values in the
        # log marginalisation results. Trying to keep them away from ~-600 or ~600
        # generally precision goes down to ~1e-300 (base 10)
        self.log_marginalisation_regularisation = np.abs(0.3*np.mean(np.diff(log_marg_maxs-log_marg_mins)))


        return nuisance_log_marg_results
    

    def select_scan_output_posterior_exploration_class(self, 
                                                       mixture_parameter_specifications: ParameterSet | list[Parameter] | dict,
                                                       mixture_fraction_exploration_type: str = None, 
                                                       log_margresults: list | np.ndarray = None,
                                                       prior_parameter_specifications: dict | list[ParameterSet] | list[dict] =None,
                                                       *args, **kwargs):
        """
        Selects and initializes (the class, not the process it contains) the appropriate exploration class based on the 
        specified mixture fraction exploration type.
        
        This method dynamically selects and initializes a class for exploring the posterior of mixture fractions. It supports
        either deterministic scanning ('scan') or stochastic sampling ('sample') methods for posterior exploration.
        
        Args:
            mixture_parameter_specifications (ParameterSet, list[Parameter], dict): Specifications for the mixture parameters involved in the exploration.
            
            mixture_fraction_exploration_type (str, optional): The type of exploration to perform. Can be 'scan' for a 
                deterministic scan or 'sample' for stochastic sampling. If not provided, defaults to the class attribute 
                `mixture_fraction_exploration_type`.
            
            log_margresults (list, array like, optional): The logarithm of marginal results to be used in the exploration. If not provided, 
                defaults to the class attribute `nuisance_log_marg_results`.
            
            prior_parameter_specifications (dict, list[ParameterSet], list[dict], optional): Specifications for prior 
            parameters involved in the exploration. If not provided, defaults to the class attribute `prior_parameter_specifications`.
            
            *args, **kwargs: Additional arguments and keyword arguments passed to the exploration class constructor.
            
        Raises:
            ValueError: If `mixture_fraction_exploration_type` is neither 'scan' nor 'sample'.
        """
        if mixture_fraction_exploration_type is None:
            mixture_fraction_exploration_type = self.mixture_fraction_exploration_type

        if prior_parameter_specifications is None:
            prior_parameter_specifications = self.prior_parameter_specifications

        if log_margresults is None:
            log_margresults = self.nuisance_log_marg_results

        if mixture_fraction_exploration_type.lower() == 'scan':
            scan_output_exploration_class = ScanOutput_ScanMixtureFracPosterior

        elif mixture_fraction_exploration_type.lower() == 'sample':
            scan_output_exploration_class = ScanOutput_StochasticMixtureFracPosterior

        else:
           raise ValueError("Invalid 'mixture_fraction_exploration_type' must be either 'scan' or 'sample'.")
        
        self.scan_output_exploration_class_instance = scan_output_exploration_class(
                log_margresults = log_margresults,
                log_nuisance_marg_regularisation = self.log_marginalisation_regularisation,
                mixture_parameter_specifications=mixture_parameter_specifications,
                prior_parameter_specifications=prior_parameter_specifications,
                *args, **kwargs)

        

    def init_posterior_exploration(self, *args, **kwargs):
        """
        Initiates the posterior exploration process.
        
        This method delegates the initiation of exploration to the instance of the exploration class selected by the 
        `select_scan_output_posterior_exploration_class` method. It prepares the exploration environment and parameters 
        based on the class instance's configuration.
        
        *args, **kwargs: Arguments and keyword arguments to be passed to the initiation method of the exploration class.
        """
        self.scan_output_exploration_class_instance.initiate_exploration(*args, **kwargs)

    def run_posterior_exploration(self, *args, **kwargs):
        """
        Runs the posterior exploration process and returns the results.
        
        This method triggers the actual exploration process using the selected and initialized exploration class instance. 
        It runs the exploration based on the configured parameters and returns the exploration results.
        
        *args, **kwargs: Arguments and keyword arguments to be passed to the run method of the exploration class.
        
        Returns:
            The results of the posterior exploration, the format and content of which depend on the exploration method used.
        """
        self._posterior_exploration_output = self.scan_output_exploration_class_instance.run_exploration(*args, **kwargs)

        return self._posterior_exploration_output


    @property
    def posterior_exploration_results(self):
        """
        Returns the results of the posterior exploration.
        
        This property provides access to the results of the posterior exploration. The nature of the results depends on the 
        mixture fraction exploration type. For 'scan', it directly returns the exploration output, while for other types, 
        it accesses the results through the sampler's `results` attribute of the exploration class instance.
        
        Returns:
            The results of the posterior exploration, which may include posterior distributions, samples, or other statistics,
            depending on the exploration type.
        """
        if self.mixture_fraction_exploration_type =='scan':
            return self._posterior_exploration_output
        else:
            return self.scan_output_exploration_class_instance.sampler.results