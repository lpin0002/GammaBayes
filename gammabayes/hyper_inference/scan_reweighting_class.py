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
from gammabayes.priors import discrete_logprior
from gammabayes.likelihoods import discrete_loglike
from gammabayes import EventData, Parameter, ParameterSet






class dynesty_restricted_proposal_marg_wrapper(object):

    def __init__(self, 
                 measured_event: list | np.ndarray, 
                 log_proposal_prior: discrete_logprior, 
                 log_likelihood: discrete_loglike, 
                 nuisance_axes: list[np.ndarray] | tuple[np.ndarray], 
                 bounds: list[list[(str, float)]]):
        
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

        if not(np.isinf(self.pseudo_log_norm)):
            flattened_logcdf = flattened_logcdf - self.pseudo_log_norm

        flattened_cdf = np.exp(flattened_logcdf)

        self.interpolation_func = interp1d(y=np.arange(flattened_cdf.size), x=flattened_cdf, 
                                    fill_value=(0, flattened_cdf.size-1), bounds_error=False)
        

        self.log_likelihood = functools.partial(
            log_likelihood.irf_loglikelihood.dynesty_single_loglikelihood,
            recon_energy=measured_event[0], recon_lon=measured_event[1], recon_lat=measured_event[2]
        )

        self.bounds = bounds

        print(f"Bounds: {self.bounds}")


    def prior_transform(self, u):
        flat_idx = self.interpolation_func(u[0])
        flat_idx = int(np.round(flat_idx))

        restricted_axis_indices = np.unravel_index(shape=self.logpdf_shape, indices=flat_idx)

        axis_values = [restricted_axis[restricted_axis_index] for restricted_axis, restricted_axis_index in zip(self.restricted_axes, restricted_axis_indices)]
        
        return axis_values
    
    def likelihood(self, x):

        log_like_val = self.log_likelihood(x)
        print(f"log_like_val: {log_like_val}")
        return log_like_val





class dynesty_scan_reweighting_class(object):

    def __init__(self, 
                 measured_events: EventData, 
                 log_likelihood: discrete_loglike, 
                 log_proposal_prior: discrete_logprior, 
                 log_target_priors: list[discrete_logprior], 
                 nuisance_axes: list[np.ndarray] = None, 
                 mixture_axes: list[np.ndarray] = None,
                 marginalisation_bounds: list[(str, float)] = None,
                 bounding_percentiles: list[float] = [90, 90],
                 bounding_sigmas: list[float] = [4,4],
                 logspace_integrator: callable = iterate_logspace_integration,
                 prior_parameter_sets: dict | list[ParameterSet] = {}, 
                 reweight_batch_size: int = 100,
                 no_priors_on_init=False):
        self.measured_events        = measured_events

        self.log_likelihood = log_likelihood
        self.log_proposal_prior = log_proposal_prior
        self.log_target_priors = log_target_priors
        self.no_priors_on_init = no_priors_on_init

        self.nuisance_axes = self._handle_nuisance_axes(nuisance_axes)
        self.mixture_axes = mixture_axes
        self.logspace_integrator = logspace_integrator

        self.prior_parameter_sets = self._handle_parameter_specification(prior_parameter_sets)
        
        self._num_parameter_specifications = len(self.prior_parameter_sets)

        self.reweight_batch_size = reweight_batch_size
        self.bounds = marginalisation_bounds


        if self.bounds is None:
            _, logeval_bound               = derive_edisp_bounds(irf_loglike=self.log_likelihood, percentile=bounding_percentiles[0], sigmalevel=bounding_sigmas[0])
            lonlact_bound                   = derive_psf_bounds(irf_loglike=self.log_likelihood, percentile=bounding_percentiles[1], sigmalevel=bounding_sigmas[1], 
                                                                axis_buffer=1, parameter_buffer = np.squeeze(np.ptp(self.dependent_axes[1]))/2)
            self.bounds = [['log10', logeval_bound], ['linear', lonlact_bound], ['linear', lonlact_bound]]

        logging.info(f"Bounds: {self.bounds}")


    def _handle_parameter_specification(self, 
                                        parameter_specifications: dict | ParameterSet,
                                        log_target_priors=None):
        """
        Processes and validates the parameter specifications provided. This method formats the input 
        parameter specifications and ensures consistency between the number of parameter specifications and the 
        number of priors.

        Args:
            parameter_specifications (dict | ParameterSet): The parameter specifications to be processed. This 
                                                            can be either a dictionary or a ParameterSet object.

        Raises:
            Exception: If the number of hyperparameter axes specified exceeds the number of priors.

        Notes:
            - A warning is issued if the number of hyperparameter axes is fewer than the number of priors. In this 
            case, empty hyperparameter axes are assigned for the missing priors.
            - If there is an issue with accessing `log_priors`, a warning is logged, and the number of priors is set 
            equal to the number of parameter specifications provided.
        """
        _num_parameter_specifications = len(parameter_specifications)
        formatted_parameter_specifications = []*_num_parameter_specifications

        if log_target_priors is None:
            log_target_priors = self.log_target_priors

        if _num_parameter_specifications>0:

            if type(parameter_specifications)==dict:

                for single_prior_parameter_specifications in parameter_specifications.items():

                    parameter_set = ParameterSet(single_prior_parameter_specifications)

                    formatted_parameter_specifications.append(parameter_set)

            elif type(parameter_specifications)==list:
                formatted_parameter_specifications = [ParameterSet(parameter_specification) for parameter_specification in parameter_specifications]

        try:
            self._num_priors = len(log_target_priors)
        except TypeError as excpt:
            logging.warning(f"An error occured when trying to calculate the number of priors: {excpt}")
            self._num_priors = _num_parameter_specifications

        if not self.no_priors_on_init or (log_target_priors is not None):

            diff_in_num_hyperaxes_vs_priors = self._num_priors-_num_parameter_specifications

            if diff_in_num_hyperaxes_vs_priors<0:
                raise Exception(f'''
You have specifed {np.abs(diff_in_num_hyperaxes_vs_priors)} more hyperparameter axes than priors.''')
            
            elif diff_in_num_hyperaxes_vs_priors>0:
                warnings.warn(f"""
You have specifed {diff_in_num_hyperaxes_vs_priors} less hyperparameter axes than priors. 
Assigning empty hyperparameter axes for remaining priors.""")
                
                _num_parameter_specifications = len(formatted_parameter_specifications)
                
                for __idx in range(_num_parameter_specifications, self._num_priors):
                    formatted_parameter_specifications.append(ParameterSet())


        return formatted_parameter_specifications

        
    def _handle_nuisance_axes(self, nuisance_axes: list[np.ndarray]):
        """
        Handles the assignment or retrieval of nuisance axes. 
        This method first checks if `nuisance_axes` is provided. If not, it attempts to retrieve nuisance axes 
        from `log_likelihood` or `log_priors`. If neither is available, it raises an exception.

        Args:
            nuisance_axes (list[np.ndarray]): A list of numpy arrays representing the nuisance axes.

        Raises:
            Exception: Raised if `nuisance_axes` is not provided and cannot be retrieved from either 
                    `log_likelihood` or `log_priors`.

        Returns:
            list[np.ndarray]: The list of numpy arrays representing the nuisance axes. This can be either the 
                            provided `nuisance_axes`, or retrieved from `log_likelihood` or `log_priors`.
        """
        if nuisance_axes is None:
            try:
                return self.log_likelihood.nuisance_axes
            except AttributeError:
                try:
                    return self.log_proposal_prior.axes
                except AttributeError:
                    try:
                        return self.log_target_priors[0].axes
                    except AttributeError:
                        raise Exception("Dependent value axes used for calculations not given.")
                    
        return nuisance_axes
            



    def run_proposal_dynesty(self, 
                             measured_event: list | np.ndarray, 
                             NestedSampler_kwargs: dict = {'nlive':200}, 
                             run_nested_kwargs: dict = {'dlogz':0.7, 'maxcall':5000}):
        
        single_event_dynesty_wrapper = dynesty_restricted_proposal_marg_wrapper(
            measured_event=measured_event, 
            log_likelihood=self.log_likelihood, 
            log_proposal_prior=self.log_proposal_prior,
            nuisance_axes=self.nuisance_axes,
            bounds = self.bounds, 
        )
    
        update_with_defaults(NestedSampler_kwargs, {'ndim': 3})

        # Set up the Nested Sampler with the multiprocessing pool
        sampler = dynesty.NestedSampler(single_event_dynesty_wrapper.log_likelihood, single_event_dynesty_wrapper.prior_transform, 
                                        **NestedSampler_kwargs)

        # Run Nested Sampling
        sampler.run_nested(**run_nested_kwargs)


        # Extract the results
        results = sampler.results

        return results
    
    def batch_process_proposal_sampling(self,
                                        measured_events: EventData, 
                                        NestedSampler_kwargs: dict = {'nlive':200}, 
                                        run_nested_kwargs: dict = {'dlogz':0.7, 'maxcall':5000}
                                        ) -> list[(float, list)]:
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

        

        
    def scan_reweight_single_prior(self, 
                                   log_evidence_values, 
                                   nuisance_parameter_samples, 
                                   proposal_prior_func,
                                   target_prior_func, 
                                   target_prior_spectral_params, 
                                   target_prior_spatial_params,
                                   target_prior_ln_norm, 
                                   proposal_prior_ln_norm, 
                                   batch_size=None):
        if batch_size is None:
            batch_size = self.reweight_batch_size
        
        

        ln_likelihood_values = []

        for log_evidence, samples in zip(log_evidence_values, nuisance_parameter_samples):
            num_samples = len(samples[:,0])
            proposal_prior_values   = np.empty(shape=(num_samples,))
            target_prior_values     = np.empty(shape=(num_samples,))

            for batch_idx in range(0, num_samples, batch_size):
                proposal_prior_values[batch_idx:batch_idx+batch_size] = proposal_prior_func(
                    samples[:,0][batch_idx:batch_idx+batch_size], 
                    samples[:,1][batch_idx:batch_idx+batch_size], 
                    samples[:,2][batch_idx:batch_idx+batch_size])
                
                target_prior_values[batch_idx:batch_idx+batch_size] = target_prior_func(
                    samples[:,0][batch_idx:batch_idx+batch_size],
                    samples[:,1][batch_idx:batch_idx+batch_size],
                    samples[:,2][batch_idx:batch_idx+batch_size],
                    spectral_parameters  = {spec_key:samples[:,0][batch_idx:batch_idx+batch_size]*0.+spec_val for spec_key, spec_val in target_prior_spectral_params.items()}, 
                    spatial_parameters   = {spat_key:samples[:,0][batch_idx:batch_idx+batch_size]*0.+spat_val for spat_key, spat_val in target_prior_spatial_params.items()})
            
            ln_likelihood_value = log_evidence-np.log(num_samples) + special.logsumexp(target_prior_values-proposal_prior_values-target_prior_ln_norm+proposal_prior_ln_norm)
            ln_likelihood_values.append(ln_likelihood_value)


        return ln_likelihood_values
    
    def scan_reweight(self, 
                      log_evidence_values, 
                      nested_sampling_results_samples, 
                      log_target_priors=None,
                      prior_parameter_sets=None):
        if log_target_priors is None:
            log_target_priors = self.log_target_priors


        if prior_parameter_sets is None:
            prior_parameter_sets = self.prior_parameter_sets
        else:
            prior_parameter_sets = self._handle_parameter_specification(prior_parameter_sets)


        normalisation_mesh = np.meshgrid(*self.nuisance_axes, indexing='ij')
        flattened_meshes = [mesh.flatten() for mesh in normalisation_mesh]




        proposal_prior_matrix= np.squeeze(
                self.log_proposal_prior(
                    *flattened_meshes, 
                    ).reshape(normalisation_mesh[0].shape))
            
        proposal_prior_ln_norm = iterate_logspace_integration(proposal_prior_matrix, axes=self.nuisance_axes)
        print(f"Proposal Ln Normalisation: {proposal_prior_ln_norm}")




        nuisance_log_marg_results = []


        for target_prior, prior_parameter_set in tqdm(zip(log_target_priors, prior_parameter_sets), total=len(log_target_priors)):
            
            prior_parameter_axis_dict = prior_parameter_set.axes_by_type

            print("\n\ntarget_prior:\n\n")
            print(target_prior)
            print(f"\nhyper_parameter_axis_dict: {prior_parameter_axis_dict}\n")

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
                spec_key:hyper_param_mesh[spec_idx] for spec_idx, spec_key in enumerate(prior_parameter_axis_dict['spectral_parameters'].keys())}
            

            hyper_param_dict_mesh['spatial_parameters'] = {
                spat_key:hyper_param_mesh[spat_idx+num_spec_parameters] for spat_idx, spat_key in enumerate(prior_parameter_axis_dict['spatial_parameters'].keys())}
            
            # Note: 0 ain't special, I'm just using it to access _a_ mesh

            try:
                example_mesh = hyper_param_mesh[0]
                num_hyper_indices = hyper_param_mesh[0].size
                example_mesh_shape = example_mesh.shape

            except:
                num_hyper_indices = 1
                example_mesh_shape = (1,)



            target_prior_log_marg_results = np.empty(shape=(len(nested_sampling_results_samples), *example_mesh_shape))


            for hyper_val_idx in tqdm(range(num_hyper_indices), total=num_hyper_indices):

                hyper_axis_indices = np.unravel_index(hyper_val_idx, example_mesh_shape)
                target_prior_matrix= np.squeeze(
                    target_prior(
                            *flattened_meshes, 
                            spectral_parameters = {
                                key:flattened_meshes[0]*0+axis[hyper_axis_indices[idx]] for idx, (key, axis) in enumerate(prior_parameter_axis_dict['spectral_parameters'].items())},
                            spatial_parameters = {
                                key:flattened_meshes[0]*0+axis[hyper_axis_indices[idx+num_spec_parameters]] for idx, (key, axis) in enumerate(prior_parameter_axis_dict['spatial_parameters'].items())},
                            ).reshape(normalisation_mesh[0].shape))
                
                target_prior_ln_norm = iterate_logspace_integration(logy=target_prior_matrix, 
                                                                        axes=self.nuisance_axes)
                if np.isinf(target_prior_ln_norm):
                    target_prior_ln_norm = 0

                single_target_prior_hyper_val_log_marg_results = np.squeeze(
                    np.asarray(
                        self.scan_reweight_single_prior(
                            log_evidence_values=log_evidence_values, 
                            nuisance_parameter_samples      = nested_sampling_results_samples,
                            target_prior_func               = target_prior, 
                            proposal_prior_func             = self.log_proposal_prior,

                            target_prior_spectral_params    = {
                                key:axis[hyper_axis_indices[idx]] for idx, (key, axis) in enumerate(prior_parameter_axis_dict['spectral_parameters'].items())},
                            target_prior_spatial_params     = {
                                key:axis[hyper_axis_indices[idx+num_spec_parameters]] for idx, (key, axis) in enumerate(prior_parameter_axis_dict['spatial_parameters'].items())},
                            
                            target_prior_ln_norm            = target_prior_ln_norm, 
                            proposal_prior_ln_norm          = proposal_prior_ln_norm
                            )
                            )
                            )

                
                target_prior_log_marg_results[:, *hyper_axis_indices] = single_target_prior_hyper_val_log_marg_results

            nuisance_log_marg_results.append(target_prior_log_marg_results)


        self.nuisance_log_marg_results = nuisance_log_marg_results


        log_marg_mins = np.asarray([np.nanmin(nuisance_log_marg_result[nuisance_log_marg_result != -np.inf]) for nuisance_log_marg_result in self.nuisance_log_marg_results])
        log_marg_maxs = np.asarray([np.nanmax(nuisance_log_marg_result[nuisance_log_marg_result != -np.inf]) for nuisance_log_marg_result in self.nuisance_log_marg_results])

        # Adaptively set the regularisation based on the range of values in the
        # log marginalisation results. Trying to keep them away from ~-600 or ~600
        # generally precision goes down to ~1e-300 (base 10)
        self.log_marginalisation_regularisation = np.abs(0.3*np.mean(np.diff(log_marg_maxs-log_marg_mins)))


        return nuisance_log_marg_results


            
    def create_discrete_mixture_log_hyper_likelihood(self, 
                                                     mixture_axes: list | tuple | np.ndarray = None, 
                                                     nuisance_log_marg_results: list | tuple | np.ndarray = None):
        
        logging.debug(f"log_marginalisation_regularisation: {self.log_marginalisation_regularisation}")
        nuisance_log_marg_results = [log_margresult + self.log_marginalisation_regularisation for log_margresult in nuisance_log_marg_results]
        
        if mixture_axes is None:
            mixture_axes = self.mixture_axes
        if nuisance_log_marg_results is None:
            nuisance_log_marg_results = self.nuisance_log_marg_results

        self.mixture_axes = mixture_axes
        mesh_mix_axes = np.meshgrid(*mixture_axes, indexing='ij')

        # To get around the fact that every component of the mixture can be a different shape
        final_output_shape = [len(nuisance_log_marg_results), *np.ones(len(mixture_axes)).astype(int), ]

        # Axes for each of the priors to __not__ expand in
        prioraxes = []

        # Counter for how many hyperparameter axes we have used
        hyper_idx = len(mixture_axes)+1


        # Creating components of mixture for each prior
        mixture_array_list = []
        for prior_idx, log_margresults_for_idx in enumerate(nuisance_log_marg_results):

            # Including 'event' and mixture axes for eventual __non__ expansion into.
                # Starting with the axis for each event (1), and the mixture axes
                # of which there are len(mesh_mix_axes)
            prior_axis_instance = list(range(1+len(mesh_mix_axes)))

            for length_of_axis in log_margresults_for_idx.shape[1:]:

                # Could potentially keep the convention of spectral axes --> spatial axes
                    # but that would require stricter requirements on the inputs to this
                    # function that aren't needed
                final_output_shape.append(length_of_axis)

                prior_axis_instance.append(hyper_idx)

                hyper_idx+=1

            prioraxes.append(prior_axis_instance)


            mixcomp = np.expand_dims(np.log(
                apply_direchlet_stick_breaking_direct(mixtures_fractions=mesh_mix_axes, depth=prior_idx)), 
                axis=(*np.delete(np.arange(log_margresults_for_idx.ndim+len(mesh_mix_axes)), 
                                    np.arange(len(mesh_mix_axes))+1),)) 


            mixcomp=mixcomp+np.expand_dims(log_margresults_for_idx, 
                                        axis=(*(np.arange(len(mesh_mix_axes))+1),)
                                    )
            

            mixture_array_list.append(mixcomp)
            
        # Making all the mixture components the same shape that is compatible for adding together (in logspace)
        for _prior_idx, mixture_array in enumerate(mixture_array_list):
            axis = []
            for _axis_idx in range(len(final_output_shape)):
                if not(_axis_idx in prioraxes[_prior_idx]):
                    axis.append(_axis_idx)
            axis = tuple(axis)

            mixture_array   = np.expand_dims(mixture_array, axis=axis)

            mixture_array_list[_prior_idx] = mixture_array

        # Now to combine the results for all the data and to 
        combined_mixture = -np.inf
        for mixture_component in mixture_array_list:
            combined_mixture = np.logaddexp(combined_mixture, mixture_component)

        log_hyperparameter_likelihood = np.sum(combined_mixture, axis=0)

        return log_hyperparameter_likelihood

