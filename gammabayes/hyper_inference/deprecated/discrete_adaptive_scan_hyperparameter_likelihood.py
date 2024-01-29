import numpy as np
import functools
from tqdm import tqdm
from gammabayes.utils import iterate_logspace_integration, logspace_riemann, update_with_defaults
from gammabayes.utils.event_axes import derive_edisp_bounds, derive_psf_bounds
from gammabayes.utils.config_utils import save_config_file
from multiprocessing import Pool
import os, warnings, logging, time

class discrete_adaptive_scan_hyperparameter_likelihood(object):
    def __init__(self, priors               = None, 
                 likelihood                 = None, 
                 axes: list[np.ndarray] | tuple[np.ndarray] | None              = None,
                 dependent_axes: list[np.ndarray] | tuple[np.ndarray] | None    = None,
                 hyperparameter_axes: dict  = {}, 
                 numcores: int              = 8, 
                 likelihoodnormalisation: np.ndarray | float                    = 0., 
                 log_margresults: np.ndarray | None                             = None, 
                 mixture_axes: list[np.ndarray] | tuple[np.ndarray] | None      = None,
                 log_hyperparameter_likelihood: np.ndarray | float              = 0., 
                 log_posterior: np.ndarray | float                              = 0., 
                 iterative_logspace_integrator: callable                        = iterate_logspace_integration,
                 logspace_integrator: callable                                  = logspace_riemann,
                 prior_matrix_list: list[np.ndarray] | tuple[np.ndarray]        = None,
                 bounds = None,
                 bounding_percentiles       = [90 ,90],
                 bounding_sigmas            = [4,4]):
        
        self.priors                     = priors
        

        self.likelihood                 = likelihood
        self.axes                       = axes
            
        if dependent_axes is None:
            try:
                self.dependent_axes             = self.likelihood.dependent_axes
            except:
                try:
                    self.dependent_axes             = self.priors[0].axes
                except:
                    raise Exception("Dependent value axes used for calculations not given.")
        else:
            self.dependent_axes             = dependent_axes

        _num_hyper_axes = len(hyperparameter_axes)

        try:
            _num_priors = len(priors)
        except Exception as excpt:
            print(f"An error occured when trying to calculate the number of priors: {excpt}")
            _num_priors = _num_hyper_axes
        
        if _num_priors==_num_hyper_axes:
            self.hyperparameter_axes        = hyperparameter_axes
        else:
            diff_in_num_hyperaxes_vs_priors = _num_priors-_num_hyper_axes
            if diff_in_num_hyperaxes_vs_priors<0:
                raise Exception(f'''
You have specifed {np.abs(diff_in_num_hyperaxes_vs_priors)} more hyperparameter axes than priors.''')
            
            else:
                warnings.warn(f"""
You have specifed {diff_in_num_hyperaxes_vs_priors} less hyperparameter axes than priors. 
Assigning empty hyperparameter axes for remaining priors.""")
                
                for __idx in range(int(diff_in_num_hyperaxes_vs_priors)):
                    hyperparameter_axes.append({'spectral_parameters':{}, 'spatial_parameters': {}})

                self.hyperparameter_axes        = hyperparameter_axes


        



        self.numcores                   = numcores
        self.likelihoodnormalisation    = likelihoodnormalisation
        self.iterative_logspace_integrator  = iterative_logspace_integrator
        self.logspace_integrator            = logspace_integrator



        self.log_margresults               = log_margresults
        if mixture_axes is None:
            self.mixture_axes               = np.array([np.linspace(0,1,101)]*(_num_priors-1))
        elif len(mixture_axes) != _num_priors-1:
            self.mixture_axes               = np.array([mixture_axes]*_num_priors)
        else:
            self.mixture_axes = np.asarray(mixture_axes)

        self.log_hyperparameter_likelihood  = log_hyperparameter_likelihood
        self.log_posterior                  = log_posterior
        self.prior_matrix_list              = prior_matrix_list

        self.bounds = bounds

        if self.bounds is None:
            _, logeval_bound               = derive_edisp_bounds(irf_loglike=self.likelihood, percentile=bounding_percentiles[0], sigmalevel=bounding_sigmas[0])
            lonlact_bound                   = derive_psf_bounds(irf_loglike=self.likelihood, percentile=bounding_percentiles[1], sigmalevel=bounding_sigmas[1], 
                                                                axis_buffer=1, parameter_buffer = np.squeeze(np.ptp(self.dependent_axes[1]))/2)
            self.bounds = [['log10', logeval_bound], ['linear', lonlact_bound], ['linear', lonlact_bound]]

        self.log_regularisation = 200

        print(f"Bounds: {self.bounds}")


    
    def bound_axis(self, axis, bound_type, bound_radii, estimated_val):
        if bound_type=='linear':
            axis_indices = np.where(
            (axis>estimated_val-bound_radii) & (axis<estimated_val+bound_radii) )[0]

        elif bound_type=='log10':
            axis_indices = np.where(
            (np.log10(axis)>np.log10(estimated_val)-bound_radii) & (np.log10(axis)<np.log10(estimated_val)+bound_radii) )[0]
            
        temp_axis = axis[axis_indices]

        return temp_axis, axis_indices

    


    def observation_nuisance_marg(self, event_vals, prior_matrix_list):



        temp_axes_and_indices = [self.bound_axis(axis, bound_type=bound_info[0], bound_radii=bound_info[1], estimated_val=event_val) for bound_info, axis, event_val in zip(self.bounds, self.likelihood.dependent_axes, event_vals)]
        temp_axes = [temp_axis_info[0] for temp_axis_info in temp_axes_and_indices]

        meshvalues  = np.meshgrid(*event_vals, *temp_axes, indexing='ij')
        index_meshes = np.ix_( *[temp_axis_info[1] for temp_axis_info in temp_axes_and_indices])

        flattened_meshvalues = [meshmatrix.flatten() for meshmatrix in meshvalues]
        
        # t2 = time.perf_counter()

        likelihoodvalues = np.squeeze(self.likelihood(*flattened_meshvalues).reshape(*meshvalues[0].shape))
        # likelihoodvalues_ndim = likelihoodvalues.ndim

        # t3 = time.perf_counter()
        likelihoodvalues = likelihoodvalues - self.likelihoodnormalisation[*index_meshes]

        # t4 = time.perf_counter()
        
        all_log_marg_results = []

        # t5_1s = []
        # t5_2s = []
        # t5_3s = []
        # t5_4s = []

        for prior_matrices in prior_matrix_list:
            # t5_1s.append(time.perf_counter())

            # Transpose is because of the convention I chose for which axes
                # were the nuisance parameters
            logintegrandvalues = (np.squeeze(prior_matrices).T[...,index_meshes[2].T,index_meshes[1].T,index_meshes[0].T]+np.squeeze(likelihoodvalues).T).T
            
            # t5_2s.append(time.perf_counter())

            single_parameter_log_margvals = iterate_logspace_integration(logintegrandvalues,   
                axes=temp_axes, axisindices=[0,1,2])

            # t5_3s.append(time.perf_counter())

            all_log_marg_results.append(np.squeeze(single_parameter_log_margvals))

            # t5_4s.append(time.perf_counter())


        # t5 = time.perf_counter()

        # t5_1s = np.array(t5_1s)
        # t5_2s = np.array(t5_2s)
        # t5_3s = np.array(t5_3s)


        # print(f"Like calc {round(t3-t2,3)}, Whole Marg Calc {round(t5-t4,3)}, Integrand Calc {round(np.mean(t5_2s-t5_1s),3)}, Integration Calc {round(np.mean(t5_3s-t5_2s),3)}, Appending Result {round(np.mean(t5_4s-t5_3s),3)}")#, end='\r')
        
        return np.array(all_log_marg_results, dtype=object)


    def process_batch(self, batch_of_event_measurements, prior_matrix_list):
        # Adjust this function to process a batch of axisvals
        results = []
        for event_measurement in batch_of_event_measurements:
            result = self.observation_nuisance_marg(event_measurement, prior_matrix_list)
            results.append(result)
        return results
    

    def split_into_batches(self, data, batch_size):
        # Implement logic to split data into batches of batch_size
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


    def prior_gen(self, Nevents: int):
        # time1 = time.perf_counter()
        # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
        np.seterr(divide='ignore', invalid='ignore')
        # log10eaxistrue,  longitudeaxistrue, latitudeaxistrue = axes
        prior_marged_shapes = np.empty(shape=(len(self.priors),), dtype=object)
        nans = 0
        prior_matrix_list = None
        if prior_matrix_list is None:
            # print("prior_matrix_list does not exist. Constructing priors.")
            prior_matrix_list = []

            for _outer_idx, prior in enumerate(self.priors):
                

                parameter_dictionary  = self.hyperparameter_axes[_outer_idx]
                update_with_defaults(parameter_dictionary, {'spectral_parameters':{}, 'spatial_parameters':{}})

                prior_spectral_params   = parameter_dictionary['spectral_parameters']
                prior_spatial_params    = parameter_dictionary['spatial_parameters']

                num_spec_params         = len(prior_spectral_params)

                prior_marged_shapes[_outer_idx] = (Nevents, 
                                                    *[prior_axis.size for prior_axis in prior_spectral_params.values()],
                                                    *[prior_axis.size for prior_axis in prior_spatial_params.values()])


                if prior.efficient_exist:
                    prior_matrices = np.squeeze(
                        prior.construct_prior_array(
                            spectral_parameters = prior_spectral_params,
                            spatial_parameters = prior_spatial_params,
                            normalise=True)
                            )
                    
                    prior_matrices = prior_matrices.reshape(*self.likelihoodnormalisation.shape, -1)
                    # print(f"prior_matrices.shape: {prior_matrices.shape}")
                    
                    nans+=np.sum(np.isnan(prior_matrices))

                else:

                    hyper_parameter_coords  = np.asarray(
                        np.meshgrid(*prior_spectral_params.values(),
                                    *prior_spatial_params.values(), indexing='ij'))
                                    
                    # Tranpose is so that when you iterate along this it accesses it by combination, not by parameter
                        # e.g. [param1_val1, parma2_val1] --> [param1_val1, parma2_val2] 
                        # vs   [param1_val1, parma1_val2] --> [param2_val1, parma2_val2] 
                    flattened_hyper_parameter_coords = np.asarray([mesh.flatten() for mesh in hyper_parameter_coords]).T

                    try:

                        prior_matrices  = np.empty(shape = (
                        *self.likelihoodnormalisation.shape,
                        len(flattened_hyper_parameter_coords))
                        )


                        prior_marged_shapes[_outer_idx] = (Nevents, *hyper_parameter_coords[0].shape)

                        

                        for _inner_idx, hyperparametervalues in tqdm(enumerate(flattened_hyper_parameter_coords), 
                                                            total=len(flattened_hyper_parameter_coords),
                                                            position=1, miniters=1, mininterval=0.1):      
                                                                        
                            prior_matrix = np.squeeze(
                                prior.construct_prior_array(
                                    spectral_parameters = {param_key: hyperparametervalues[param_idx] for param_idx, param_key in enumerate(prior_spectral_params.keys())},
                                    spatial_parameters = {param_key: hyperparametervalues[num_spec_params+ param_idx] for param_idx, param_key in enumerate(prior_spatial_params.keys())},
                                    normalise=True)
                                    )
                            
                            prior_matrix = prior_matrix #- iterate_logspace_integration(logy=prior_matrix, 
                                                                                    #    axes=self.dependent_axes)
                            

                            print(prior_matrix.shape)
                            nans+=np.sum(np.isnan(prior_matrix))
                            prior_matrices[...,_inner_idx] = prior_matrix
                        


                    except (TypeError, IndexError) as err:
                        # warnings.warn(f'No hyperparameters axes specified for prior number {_outer_idx}.')
                        logging.info(f"An error occurred: {err}")

                        prior_matrices  = np.empty(shape = (
                        1,
                        *np.squeeze(self.likelihoodnormalisation).shape,)
                        )

                        prior_marged_shapes[_outer_idx] = (Nevents, )

                        prior_matrix = np.squeeze(
                                prior.construct_prior_array(
                                    spectral_parameters = {},
                                    spatial_parameters = {},
                                    normalise=True)
                                    )
            
                        prior_matrices[0,...] = prior_matrix


                prior_matrices = np.asarray(prior_matrices, dtype=float)

                prior_matrices.reshape(*self.likelihoodnormalisation.shape, -1)


                prior_matrix_list.append(prior_matrices)


            # print(f"Total cumulative number of nan values within all prior matrices: {nans}")
                
            self.prior_matrix_list = prior_matrix_list

        # time2 = time.perf_counter()
        # print(f"Prior Matrix Gen Time: {time2-time1}")
            
        return prior_marged_shapes, prior_matrix_list
            

    def nuisance_log_marginalisation(self, event_vals: list | np.ndarray) -> list:

        prior_marged_shapes, prior_matrix_list = self.prior_gen(Nevents=len(event_vals))

        marg_partial = functools.partial(
            self.process_batch, 
            prior_matrix_list=self.prior_matrix_list)
        

        batched_event_measurements = self.split_into_batches(event_vals, 10)

        with Pool(self.numcores) as pool:
            marg_results = pool.map(marg_partial, batched_event_measurements)


        marg_results = [item for sublist in marg_results for item in sublist]

        # time3 = time.perf_counter()
        # print(f"Marg Time: {time3-time2}")

        marg_results = np.asarray(marg_results)

        # print(marg_results.shape)

        reshaped_marg_results = []
        for _prior_idx in range(len(self.priors)):
            nice_marg_results = np.squeeze(np.vstack(marg_results[:,_prior_idx]))
            # print('\n')
            # print('nice shape: ', nice_marg_results.shape)
            nice_marg_results = nice_marg_results.reshape(*prior_marged_shapes[_prior_idx])
            reshaped_marg_results.append(nice_marg_results)

        self.log_regularisation = np.mean([0.25*np.nanmin(reshaped_marg_result[reshaped_marg_result != -np.inf]) for reshaped_marg_result in reshaped_marg_results])

        return reshaped_marg_results

    def apply_direchlet_stick_breaking_direct(self, 
                                              xi_axes: list | tuple, 
                                              depth: int) -> np.ndarray | float:
        direchletmesh = 1

        for _dirichlet_i in range(depth):
            direchletmesh*=(1-xi_axes[_dirichlet_i])
        if depth!=len(xi_axes):
            direchletmesh*=xi_axes[depth]

        return direchletmesh
    
    
    
    def add_log_nuisance_marg_results(self, new_log_marg_results: np.ndarray) -> None:
        """Add log nuisance marginalisation results to those within the class.

        Args:
            new_log_marg_results (np.ndarray): The log likelihood values after 
                marginalising over nuisance parameters.
        """
        self.log_margresults = np.append(self.log_margresults, new_log_marg_results, axis=0)
            
    def _vector_process_create_discrete_mixture_log_hyper_likelihood(self, 
                                                            log_margresults: list | tuple | np.ndarray = None,
                                                            mixture_axes: list | tuple | np.ndarray = None, 
                                                            
                                                            ):
        if mixture_axes is None:
            mixture_axes = self.mixture_axes
            if mixture_axes is None:
                raise Exception("Mixture axes not specified")
        else:
            self.mixture_axes = mixture_axes
        if not(self.priors is None):
            if len(mixture_axes)!=len(self.priors)-1:
                raise Exception(f""""Number of mixture axes does not match number of components (minus 1). Please check your inputs.
Number of mixture axes is {len(mixture_axes)}
and number of prior components is {len(self.priors)}.""")
        if log_margresults is None:
            log_margresults = self.log_margresults

        mix_axes = np.meshgrid(*mixture_axes, indexing='ij')

        # To get around the fact that every component of the mixture can be a different shape
        final_output_shape = [len(log_margresults), *np.arange(len(mixture_axes)), ]

        # Axes for each of the priors to __not__ expand in
        prioraxes = []

        # Counter for how many hyperparameter axes we have used
        hyper_idx = len(mixture_axes)+1


        # Creating components of mixture for each prior
        mixture_array_comp_list = []
        for prior_idx, log_margresults_for_idx in enumerate(log_margresults):

            # Including 'event' and mixture axes for eventual __non__ expansion into
            prior_axis_instance = list(range(1+len(mix_axes)))

            for length_of_axis in log_margresults_for_idx.shape[1:]:

                final_output_shape.append(length_of_axis)

                prior_axis_instance.append(hyper_idx)

                hyper_idx+=1

            prioraxes.append(prior_axis_instance)


            mixcomp = np.expand_dims(np.log(
                self.apply_direchlet_stick_breaking_direct(xi_axes=mix_axes, depth=prior_idx)), 
                axis=(*np.delete(np.arange(log_margresults_for_idx.ndim+len(mix_axes)), 
                                    np.arange(len(mix_axes))+1),)) 


            mixcomp=mixcomp+np.expand_dims(log_margresults_for_idx, 
                                        axis=(*(np.arange(len(mix_axes))+1),)
                                    )
            

            mixture_array_comp_list.append(mixcomp)
            
        # Making all the mixture components the same shape that is compatible for adding together (in logspace)
        for _prior_idx, mixture_array in enumerate(mixture_array_comp_list):
            axis = []
            for _axis_idx in range(len(final_output_shape)):
                if not(_axis_idx in prioraxes[_prior_idx]):
                    axis.append(_axis_idx)
            axis = tuple(axis)

            mixture_array   = np.expand_dims(mixture_array, axis=axis)

            mixture_array_comp_list[_prior_idx] = mixture_array

        # Now to combine the results for all the data and to 
        combined_mixture = -np.inf
        for mixture_component in mixture_array_comp_list:
            combined_mixture = np.logaddexp(combined_mixture, mixture_component)

        log_hyperparameter_likelihood = np.sum(combined_mixture, axis=0)

        return log_hyperparameter_likelihood
    
    def _batch_create_discrete_mixture_log_hyper_likelihood(self, 
                                                            batched_log_margresults: list | tuple | np.ndarray = None,
                                                            mixture_axes: list | tuple | np.ndarray = None, 
                                                            mixture_buffer: int = 10
                                                            ):    
        


         # self.log_hyper_param_likelihood =  0
        # for _skip_idx in tqdm(range(int(float(self.config_dict['Nevents']/self.mixture_scanning_buffer)))):
        #     _temp_log_marg_reults = [self.margresults[_index][_skip_idx*self.mixture_scanning_buffer:_skip_idx*self.mixture_scanning_buffer+self.mixture_scanning_buffer, ...] for _index in range(len(self.margresults))]
        #     self.log_hyper_param_likelihood = self.hyperparameter_likelihood_instance.update_hyperparameter_likelihood(self.hyperparameter_likelihood_instance.create_discrete_mixture_log_hyper_likelihood(
        #         mixture_axes=mixtureaxes, log_margresults=_temp_log_marg_reults))

        batched_log_hyper_param_likelihood= 0
        for _batch_idx in range(0, len(batched_log_margresults[0]), mixture_buffer):
            batched_batched_log_margresults = [batched_log_margresults[_prior_index][_batch_idx:_batch_idx+mixture_buffer, ...] for _prior_index in range(len(batched_log_margresults))]
            batched_log_hyper_param_likelihood += self._vector_process_create_discrete_mixture_log_hyper_likelihood(
                                                            log_margresults=batched_batched_log_margresults,
                                                            mixture_axes=mixture_axes,
                                                            )
            
        return batched_log_hyper_param_likelihood
            

    def create_discrete_mixture_log_hyper_likelihood(self, 
                                                     mixture_axes: list | tuple | np.ndarray = None, 
                                                     log_margresults: list | tuple | np.ndarray = None,
                                                     num_in_mixture_batch: int = 10,
                                                     mixture_buffer: int = 10):
        
        self.mixture_axes = mixture_axes
        # batched_log_margresults = self.split_into_batches(log_margresults, num_in_mixture_batch)

        log_margresults = [logmarg + self.log_regularisation for logmarg in log_margresults]
        batched_log_margresults = [
            [log_margresults[_index][_skip_idx:_skip_idx+num_in_mixture_batch, ...] for _index in range(len(log_margresults))]
            for _skip_idx in range(0, len(log_margresults[0]), num_in_mixture_batch)]

        create_mixture_partial = functools.partial(
            self._batch_create_discrete_mixture_log_hyper_likelihood, 
            mixture_axes=mixture_axes, mixture_buffer = mixture_buffer) 
                

        log_hyper_param_likelihood = self.log_hyperparameter_likelihood

        with Pool(self.numcores) as pool:
            # We use map_async here as all the results will be multiplied (summed in logspace) in the end anyways
                # and generally aynchronous process management is faster than synchronous
            for result in pool.imap(create_mixture_partial, batched_log_margresults):
                log_hyper_param_likelihood += result


        # log_hyper_param_likelihood += np.sum(create_mixture_results, axis=0)

        self.log_hyperparameter_likelihood = log_hyper_param_likelihood - self.log_regularisation

        return log_hyper_param_likelihood
        

    def update_hyperparameter_likelihood(self, 
                                           log_hyperparameter_likelihood: np.ndarray
                                           ) -> np.ndarray:
        """To combine log hyperparameter likelihoods from multiple runs by 
            adding the resultant log_hyperparameter_likelihood together.

        Args:
            log_hyperparameter_likelihood (np.ndarray): Hyperparameter 
                log-likelihood results from a separate run with the same hyperparameter axes.

        Returns:
            np.ndarray: Updated log_hyperparameter_likelihood
        """
        self.log_hyperparameter_likelihood += log_hyperparameter_likelihood

        return self.log_hyperparameter_likelihood

    def apply_uniform_hyperparameter_priors(self, priorinfos: list[dict] | tuple[dict], 
                                            hyper_param_axes: list[np.ndarray] | tuple[np.ndarray] | None = None, 
                                            log_hyper_priormesh: np.ndarray = None, 
                                            integrator: callable = None):
        """A function to apply uniform log priors for the given prior information.

        Format for each set of prior information within the tuple should be
        priorinfo['min'] = minimum value of hyperparameter range
        priorinfo['max'] = maximum value of hyperparameter range
        priorinfo['spacing'] = hyperparameter value spacing (log10/linear)
        priorinfo['uniformity'] = Whether prior values are uniform (linear) 
                                                    or log-uniform (log) 
        priorinfo['bins] = number of entries sampled within the hyperparameter range

        Optional:
            priorinfo['upper_cutoff'] = Cut-off value above which values aren't used.
                                        Defaults to max value.
            priorinfo['lower_cutoff'] = Cut-off value below which values aren't used
                                        Defaults to min value.
            Used as `hyper_param_axis[np.where(np.logical_and(hyper_param_axis>=lower_cutoff, 
                                                              hyper_param_axis<=upper_cutoff,))]
    
        And each entry of the tuple then corresponds to each hyperparameter
        Args:
            priorinfo (tuple): A tuple containing the min, max and spacing
                of the hyperparameter axes sampled to generate the discrete 
                hyperparameter log likelihood

            hyper_param_axes (tuple of array-like): A tuple containing the values of
                the priors. If given min, max and bin arguments are not needed in priorinfo

            log_hyper_priormesh (float or array-like): An array or float presenting 
                wanted mesh of log prior values. Overwrites priorinfo and 
                hyper_param_axes input if given
        Returns:
            array like: The natural log of the discrete hyperparameter uniform 
                prior values
        """
        if integrator is None:
            integrator = self.logspace_integrator
        if log_hyper_priormesh is None:
            hyper_val_list = []
            log_prior_val_list = []
            axes_included_indices = []
            if hyper_param_axes is None:
                for prior_info in priorinfos:
                    print('prior_info: ', prior_info)

                    if prior_info['spacing'].lower()=='log':
                        hyper_param_axis = np.logspace(np.log(prior_info['min']), np.log(prior_info['max']), prior_info['bins'])
                        hyper_param_axes.append(hyper_param_axis)

                    elif prior_info['spacing'].lower()=='linear':
                        hyper_param_axis = np.linspace(prior_info['min'], prior_info['max'], prior_info['bins'])
                        hyper_param_axes.append(hyper_param_axis)


            for prior_info, hyper_param_axis in zip(priorinfos, hyper_param_axes):
                print('There is a hyper param axis')
                if prior_info['uniformity'].lower()=='log' and prior_info['spacing'].lower()=='log10':
                    log_hyper_prior_axis = hyper_param_axis*0 # Just need a constant value
                    log_hyper_prior_axis = log_hyper_prior_axis-integrator(logy=log_hyper_prior_axis, x=np.log10(hyper_param_axis))

                elif prior_info['uniformity'].lower()=='log' and not(prior_info['spacing'].lower()=='log10'):
                    log_hyper_prior_axis = np.log(1/hyper_param_axis)# Just need something proportional to 1/hyperparameter value
                    log_hyper_prior_axis = log_hyper_prior_axis-integrator(logy=log_hyper_prior_axis, x=hyper_param_axis)

                elif prior_info['uniformity'].lower()=='linear' and prior_info['spacing'].lower()=='log10':
                    log_hyper_prior_axis = np.log(hyper_param_axis)# Just need values proportional to the hyperparameter values
                    log_hyper_prior_axis = log_hyper_prior_axis-integrator(logy=log_hyper_prior_axis, x=hyper_param_axis)

                else:
                    log_hyper_prior_axis = hyper_param_axis*0# Just need a constant value
                    log_hyper_prior_axis = log_hyper_prior_axis-integrator(logy=log_hyper_prior_axis, x=hyper_param_axis)


                lower_cutoff_val = prior_info.get('lower_cutoff', hyper_param_axis.min())
                upper_cutoff_val = prior_info.get('upper_cutoff', hyper_param_axis.max())

                axes_included_indices.append(np.where(np.logical_and(hyper_param_axis>=lower_cutoff_val, 
                                                                            hyper_param_axis<=upper_cutoff_val,)))

                hyper_val_list.append(hyper_param_axis)
                log_prior_val_list.append(log_hyper_prior_axis)
            print(axes_included_indices)


        for idx, axes_included_indexs in enumerate(axes_included_indices):
            log_prior_val_list[idx] = log_prior_val_list[idx][axes_included_indexs]
            hyper_val_list[idx] = hyper_val_list[idx][axes_included_indexs]


        log_hyper_priormesh_list = np.meshgrid(*log_prior_val_list, indexing='ij')
        log_hyper_priormesh = np.prod(log_hyper_priormesh_list, axis=0)

        self.log_posterior = np.squeeze(self.log_hyperparameter_likelihood)+log_hyper_priormesh

        return self.log_posterior, log_prior_val_list, hyper_val_list

            
            
    def pack_data(self, reduce_mem_consumption: bool = True):
        """A method to pack the information contained within a class 
        instance into a dictionary.

        This method saves all the attributes of a class instance to a dictionary. 
            By default the input priors and likelihoods are not saved to save due
            to the large amount of memory they can consume. If this is not an 
            issue, and you wish to save these then set reduce_data_consumption to 
            False.

        Args:

            reduce_mem_consumption (bool, optional): A bool parameter of whether
                to not save the input prior and likelihoods to save on memory 
                consumption. Defaults to True (meaning to __not__ save the prior 
                and likelihood).
        """
        packed_data = {}
        
        packed_data['hyperparameter_axes']                 = self.hyperparameter_axes
        if not(reduce_mem_consumption):
            packed_data['priors']                              = self.priors
            packed_data['likelihood']                          = self.likelihood
            
        packed_data['axes']                                = self.axes
        packed_data['dependent_axes']                      = self.dependent_axes
        packed_data['log_hyperparameter_likelihood']       = self.log_hyperparameter_likelihood
        packed_data['mixture_axes']                        = self.mixture_axes

        packed_data['log_posterior']                        = self.log_posterior
                
        return packed_data
    

    def save_data(self, file_name="hyper_class_data.yaml", **kwargs):

        data_to_save = self.pack_data(**kwargs)
        try:
            save_config_file(data_to_save, file_name)
        except:
            warnings.warn("""Something went wrong when trying to save to the specified directory. 
Saving to current working directory as hyper_class_data.yaml""")
            save_config_file("hyper_class_data.yaml", file_name)
