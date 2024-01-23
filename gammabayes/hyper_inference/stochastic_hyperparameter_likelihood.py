from scipy import special
from scipy.interpolate import interp1d
from tqdm import tqdm
import numpy as np
import functools, dynesty, warnings, os, sys, time
from matplotlib import pyplot as plt
from gammabayes.utils.event_axes import derive_edisp_bounds, derive_psf_bounds
from gammabayes.utils import update_with_defaults, iterate_logspace_integration
from gammabayes.samplers.sampler_utils import dynesty_restricted_proposal_marg_wrapper



class dynesty_scan_reweighting_class(object):

    def __init__(self, measured_events, irf_loglike, proposal_prior, target_priors, 
                 nuisance_axes, mixture_axes=None,
                 logenergy_bound_percentile=90, logenergy_bound_sigmalevel=4, logenergy_bound=None, 
                 lonlat_bound_percentile=68, lonlat_bound_sigmalevel=4, lonlat_bound=None,
                 logspace_integrator=iterate_logspace_integration,
                 hyper_parameter_axis_dictionaries={}, reweight_batch_size=100):
        self.measured_events        = measured_events
        self.irf_loglike            = irf_loglike
        self.proposal_prior         = proposal_prior
        self.target_priors          = target_priors
        self.nuisance_axes          = nuisance_axes
        self.mixture_axes           = mixture_axes
        self.logspace_integrator    = logspace_integrator
        self.hyper_parameter_axis_dictionaries = hyper_parameter_axis_dictionaries
        self.reweight_batch_size    = reweight_batch_size


        self.logenergy_bound = logenergy_bound
        self.lonlat_bound = lonlat_bound

        if self.logenergy_bound is None:
            self.logenergy_bound    = derive_edisp_bounds(irf_loglike=irf_loglike, percentile=logenergy_bound_percentile, sigmalevel=logenergy_bound_sigmalevel)            


        if self.lonlat_bound is None:
            self.lonlat_bound       = derive_psf_bounds(irf_loglike=irf_loglike, percentile=lonlat_bound_percentile, sigmalevel=lonlat_bound_sigmalevel)
            


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
        print(log_evidence[-1])

        return log_evidence[-1], results.samples

        
    def scan_reweight_single_prior(self, log_evidence_values, nuisance_parameter_samples, proposal_prior_func,
                    target_prior_func, target_prior_spectral_params, target_prior_spatial_params,
                    target_prior_ln_norm, proposal_prior_ln_norm, batch_size=None):
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
                      log_evidence_values, nested_sampling_results_samples, 
                      target_priors=None,
                      hyper_parameter_axis_dictionaries=None):
        if target_priors is None:
            target_priors = self.target_priors


        if hyper_parameter_axis_dictionaries is None:
            hyper_parameter_axis_dictionaries = [{} for target_prior in target_priors]


        if len(hyper_parameter_axis_dictionaries)!= len(target_priors):
            for addition_idx in range(len(hyper_parameter_axis_dictionaries), len(target_priors)):
                hyper_parameter_axis_dictionaries.append({})

        normalisation_mesh = np.meshgrid(*self.nuisance_axes, indexing='ij')
        flattened_meshes = [mesh.flatten() for mesh in normalisation_mesh]




        proposal_prior_matrix= np.squeeze(
                self.proposal_prior(
                    *flattened_meshes, 
                    ).reshape(normalisation_mesh[0].shape))
            
        proposal_prior_ln_norm = iterate_logspace_integration(proposal_prior_matrix, axes=self.nuisance_axes)
        print(f"Proposal Ln Normalisation: {proposal_prior_ln_norm}")




        nuisance_log_marg_results = []


        for target_prior, hyper_parameter_axis_dict in tqdm(zip(target_priors, hyper_parameter_axis_dictionaries), total=len(target_priors)):

            update_with_defaults(hyper_parameter_axis_dict, 
                                 {'spectral_parameters':{}, 
                                  'spatial_parameters':{}})
            hyper_param_mesh = np.meshgrid(
                *hyper_parameter_axis_dict['spectral_parameters'].values(),
                *hyper_parameter_axis_dict['spatial_parameters'].values(),
                indexing='ij')
            
            num_spec_parameters = len(hyper_parameter_axis_dict['spectral_parameters'].keys())


            hyper_param_dict_mesh = {}
            hyper_param_dict_mesh['spectral_parameters'] = {
                spec_key:hyper_param_mesh[spec_idx] for spec_idx, spec_key in enumerate(hyper_parameter_axis_dict['spectral_parameters'].keys())}
            

            hyper_param_dict_mesh['spatial_parameters'] = {
                spat_key:hyper_param_mesh[spat_idx+num_spec_parameters] for spat_idx, spat_key in enumerate(hyper_parameter_axis_dict['spatial_parameters'].keys())}
            
            # Note: 0 ain't special, I'm just using it to access _a_ mesh

            try:
                example_mesh = hyper_param_mesh[0]
                num_hyper_indices = len(hyper_param_mesh[0])
                example_mesh_shape = example_mesh.shape

            except:
                num_hyper_indices = 1
                example_mesh_shape = (1,)



            target_prior_log_marg_results = np.empty(shape=(len(nested_sampling_results_samples), *example_mesh_shape))


            for hyper_val_idx in range(num_hyper_indices):

                hyper_axis_indices = np.unravel_index(hyper_val_idx, example_mesh_shape)
                target_prior_matrix= np.squeeze(
                    target_prior(
                            *flattened_meshes, 
                            spectral_parameters = {
                                key:flattened_meshes[0]*0+axis[hyper_axis_indices[idx]] for idx, (key, axis) in enumerate(hyper_parameter_axis_dict['spectral_parameters'].items())},
                            spatial_parameters = {
                                key:flattened_meshes[0]*0+axis[hyper_axis_indices[idx+num_spec_parameters]] for idx, (key, axis) in enumerate(hyper_parameter_axis_dict['spatial_parameters'].items())},
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
                            proposal_prior_func             = self.proposal_prior,

                            target_prior_spectral_params    = {
                                key:axis[hyper_axis_indices[idx]] for idx, (key, axis) in enumerate(hyper_parameter_axis_dict['spectral_parameters'].items())},
                            target_prior_spatial_params     = {
                                key:axis[hyper_axis_indices[idx+num_spec_parameters]] for idx, (key, axis) in enumerate(hyper_parameter_axis_dict['spatial_parameters'].items())},
                            
                            target_prior_ln_norm            = target_prior_ln_norm, 
                            proposal_prior_ln_norm          = proposal_prior_ln_norm
                            )
                            )
                            )

                
                target_prior_log_marg_results[:, *hyper_axis_indices] = single_target_prior_hyper_val_log_marg_results

            nuisance_log_marg_results.append(target_prior_log_marg_results)


        self.nuisance_log_marg_results = nuisance_log_marg_results


        return nuisance_log_marg_results



    def apply_direchlet_stick_breaking_direct(self, xi_axes: list | tuple, 
                                            depth: int) -> np.ndarray | float:
        direchletmesh = 1

        for _dirichlet_i in range(depth):
            direchletmesh*=(1-xi_axes[_dirichlet_i])
        if depth!=len(xi_axes):
            direchletmesh*=xi_axes[depth]

        return direchletmesh

            
    def create_discrete_mixture_log_hyper_likelihood(self, mixture_axes: list | tuple | np.ndarray = None, 
                                                    nuisance_log_marg_results: list | tuple | np.ndarray = None):
        
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
                self.apply_direchlet_stick_breaking_direct(xi_axes=mesh_mix_axes, depth=prior_idx)), 
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

