import numpy as np
import functools
from tqdm import tqdm
from gammabayes.utils import iterate_logspace_integration, logspace_riemann, update_with_defaults
from gammabayes.utils.config_utils import save_config_file
from gammabayes import EventData, Parameter, ParameterSet
from gammabayes.priors import discrete_logprior
from multiprocessing.pool import ThreadPool as Pool
import os, warnings, logging, time

class discrete_brute_scan_hyperparameter_likelihood(object):
    def __init__(self, priors: list[discrete_logprior] | tuple[discrete_logprior] = None, 
                 likelihood: callable = None, 
                 axes: list[np.ndarray] | tuple[np.ndarray] | None=None,
                 dependent_axes: list[np.ndarray] | tuple[np.ndarray] | None = None,
                 parameter_specifications: dict = {}, 
                 numcores: int = 8, 
                 likelihoodnormalisation: np.ndarray | float = 0., 
                 log_margresults: np.ndarray | None = None, 
                 mixture_param_specifications: list[np.ndarray] | tuple[np.ndarray] | None = None,
                 log_hyperparameter_likelihood: np.ndarray | float = 0., 
                 log_posterior: np.ndarray | float = 0., 
                 iterative_logspace_integrator: callable = iterate_logspace_integration,
                 logspace_integrator: callable = logspace_riemann,
                 prior_matrix_list: list[np.ndarray] | tuple[np.ndarray] = None):
        """Initialise a hyperparameter_likelihood class instance.

        Args:
            priors (tuple|list, optional): Tuple containing instances of the 
                discrete_logprior object. Defaults to None.

            likelihood (function, optional): A function that takes in axes and 
                dependent axes and output the relevant log-likelihood values. 
                Defaults to None.

            axes (tuple, optional): Tuple of np.ndarrays representing the 
                possible values that measurements can take. e.g. axis of 
                possible energy values, longitude values, and latitude values. 
                Defaults to None.

            dependent_axes (tuple, optional): Tuple of np.ndarrays 
                representing the possible values that true values of gamma-ray
                events can take. e.g. axis of possible true energy values, true
                longitude values, and true latitude values. 
                Defaults to None.

            hyperparameter_axes (dict, optional): Dict containing the default 
                values at which the priors will be evaluated. Defaults to {}.

            numcores (int, optional): If wanting to multi-core parallelize, 
                this represents the number of cores that will be used. 
                Defaults to 8.

            likelihoodnormalisation (np.ndarray or float, optional): An array
                containing the log of the normalisation values of the 
                likelihoods if interpolation method within function does not
                preserve normalisation (likely). Defaults to ().

            log_marg_results (np.ndarray, optional): Log of nuisance 
                marginalisation results. Defaults to None.

            mixture_axes (tuple, optional): A tuple containing the weights for
                each prior within a mixture model. Defaults to None.

            log_hyperparameter_likelihood (np.ndarray, optional): A numpy array
                containing the log of hyperparameter marginalisation results. 
                Defaults to 0.

            log_posterior (np.ndarray, optional): A numpy array containing the
                log of hyperparameter posterior results. Defaults to 0.
        """
        
        self.priors             = priors
        

        self.likelihood         = likelihood
        self.axes               = axes
            
        self.dependent_axes     = self._handle_dependent_axes(dependent_axes)


        self._handle_parameter_specification(parameter_specifications=parameter_specifications)

        self.numcores                       = numcores
        self.likelihoodnormalisation        = np.asarray(likelihoodnormalisation)
        self.iterative_logspace_integrator  = iterative_logspace_integrator
        self.logspace_integrator            = logspace_integrator



        self.log_margresults                = log_margresults

        self.mixture_param_set              = ParameterSet(mixture_param_specifications)

        self.log_hyperparameter_likelihood  = log_hyperparameter_likelihood
        self.log_posterior                  = log_posterior
        self.prior_matrix_list              = prior_matrix_list


    def _handle_dependent_axes(self, dependent_axes):
        if dependent_axes is None:
            try:
                return self.likelihood.dependent_axes
            except AttributeError:
                try:
                    return self.priors[0].axes
                except AttributeError:
                    raise Exception("Dependent value axes used for calculations not given.")
        return dependent_axes


    def _handle_parameter_specification(self, parameter_specifications):
        _num_parameter_specifications = len(parameter_specifications)
        formatted_parameter_specifications = []*_num_parameter_specifications

        for single_prior_parameter_specifications in parameter_specifications.items():

            parameter_set = ParameterSet(single_prior_parameter_specifications)

            formatted_parameter_specifications.append(parameter_set)

        try:
            self._num_priors = len(self.priors)
        except TypeError as excpt:
            logging.warn(f"An error occured when trying to calculate the number of priors: {excpt}")
            self._num_priors = _num_parameter_specifications
        

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

        self._num_parameter_specifications  = len(formatted_parameter_specifications)
        self.parameter_specifications       = formatted_parameter_specifications
        



    
    def observation_nuisance_marg(self, 
                                  axisvals: list | np.ndarray, 
                                  prior_matrix_list: list[np.ndarray] | tuple[np.ndarray]) -> np.ndarray:
        """Returns a list of the log marginalisation values for a single set of gamma-ray
            event measurements for various log prior matrices.

        
        
        This function takes in a single set of observation values to create the
            log of the marginalisation result over the nuisance parameters,
            otherwise known as the dependent axes or true values, for a set of
            signal log prior matrices and single background log prior matrix over
            the dependent axes.


        Args:
            axisvals (tuple): A tuple of the set of measurements for a single
                gamma-ray event.

            prior_matrix_list (list): A list of lists that contain the log prior 
                matrices for the various values of the relevant hyperparameters 
                for eeach prior.

        Returns:
            np.ndarray: A numpy array of the log marginalisation values for 
                each of the priors.
        """

        meshvalues  = np.meshgrid(*axisvals, *self.dependent_axes, indexing='ij')
        flattened_meshvalues = [meshmatrix.flatten() for meshmatrix in meshvalues]
        
        t2 = time.perf_counter()

        likelihoodvalues = np.squeeze(self.likelihood(*flattened_meshvalues).reshape(meshvalues[0].shape))

        t3 = time.perf_counter()

        likelihoodvalues = likelihoodvalues - self.likelihoodnormalisation

        t4 = time.perf_counter()
        
        all_log_marg_results = []

        t5_1s = []
        t5_2s = []
        t5_3s = []
        t5_4s = []

        for prior_matrices in prior_matrix_list:
            t5_1s.append(time.perf_counter())

            # Transpose is because of the convention I chose for which axes
                # were the nuisance parameters
            logintegrandvalues = (np.squeeze(prior_matrices).T+np.squeeze(likelihoodvalues).T).T
            
            t5_2s.append(time.perf_counter())

            single_parameter_log_margvals = self.iterative_logspace_integrator(logintegrandvalues,   
                axes=self.dependent_axes)

            t5_3s.append(time.perf_counter())

            all_log_marg_results.append(np.squeeze(single_parameter_log_margvals))

            t5_4s.append(time.perf_counter())


        t5 = time.perf_counter()

        t5_1s = np.array(t5_1s)
        t5_2s = np.array(t5_2s)
        t5_3s = np.array(t5_3s)


        logging.debug(f"Like calc {round(t3-t2,3)}, Whole Marg Calc {round(t5-t4,3)}, Integrand Calc {round(np.mean(t5_2s-t5_1s),3)}, Integration Calc {round(np.mean(t5_3s-t5_2s),3)}, Appending Result {round(np.mean(t5_4s-t5_3s),3)}")
        
        return np.array(all_log_marg_results, dtype=object)


    def prior_gen(self, Nevents: int) -> list:

        
        # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
        np.seterr(divide='ignore', invalid='ignore')
        # log10eaxistrue,  longitudeaxistrue, latitudeaxistrue = axes
        prior_marged_shapes = np.empty(shape=(len(self.priors),), dtype=object)
        nans = 0
        if self.prior_matrix_list is None:
            logging.info("prior_matrix_list does not exist. Constructing priors.")
            prior_matrix_list = []

            for _prior_idx, prior in tqdm(enumerate(self.priors), 
                                   total=len(self.priors), 
                                   desc='Setting up prior matrices'):
                

                prior_parameter_specifications = self.parameter_specifications[_prior_idx].scan_format

                prior_spectral_params   = prior_parameter_specifications['spectral_parameters']
                prior_spatial_params    = prior_parameter_specifications['spatial_parameters']


                prior_marged_shapes[_prior_idx] = (Nevents, 
                                                   *[parameter_specification.size for parameter_specification in prior_spectral_params.values()],
                                                   *[parameter_specification.size for parameter_specification in prior_spatial_params.values()])


                if prior.efficient_exist:
                    prior_matrices = np.squeeze(
                        prior.construct_prior_array(
                            spectral_parameters = prior_spectral_params,
                            spatial_parameters =  prior_spatial_params,
                            normalise=True)
                            )
                    
                    prior_matrices = prior_matrices.reshape(*self.likelihoodnormalisation.shape, -1)
                    
                    nans+=np.sum(np.isnan(prior_matrices))

                else:

                    prior_matrices, nans = self._mesh_inefficient_prior_construction( 
                                             prior=prior, prior_idx=_prior_idx,
                                             prior_spectral_params=prior_spectral_params, 
                                             prior_spatial_params=prior_spatial_params, 
                                             Nevents=Nevents, prior_marged_shapes=prior_marged_shapes,
                                             nans=nans
                                             )


                prior_matrices = np.asarray(prior_matrices, dtype=float)

                prior_matrix_list.append(prior_matrices)

            logging.debug(f"Total cumulative number of nan values within all prior matrices: {nans}")
                
            self.prior_matrix_list = prior_matrix_list

            return prior_marged_shapes, prior_matrix_list
        

    def _mesh_inefficient_prior_construction(self, 
                                             prior, prior_idx,
                                             prior_spectral_params, prior_spatial_params, 
                                             Nevents, prior_marged_shapes,
                                             nans
                                             ):
        
        num_spec_params         = len(prior_spectral_params)


        hyper_parameter_coords  = np.asarray(
                        np.meshgrid(*[parameter_specification for parameter_specification in prior_spectral_params.values()],
                                    *[parameter_specification for parameter_specification in prior_spatial_params.values()], indexing='ij'))
                                    

        flattened_hyper_parameter_coords = np.asarray([mesh.flatten() for mesh in hyper_parameter_coords]).T

        not_empty_params = len(prior_spectral_params) + len(prior_spatial_params)

        if not_empty_params:
            num_prior_values = flattened_hyper_parameter_coords.shape[0]
            parameter_matrix_shape = hyper_parameter_coords[0].shape

        else:
            num_prior_values = 1
            parameter_matrix_shape = ()
            flattened_hyper_parameter_coords = [[]]


        prior_matrices  = np.empty(shape = (
        num_prior_values,
        *np.squeeze(self.likelihoodnormalisation).shape,)
        )


        prior_marged_shapes[prior_idx] = (Nevents, *parameter_matrix_shape)

        

        for _inner_idx, hyperparametervalues in tqdm(enumerate(flattened_hyper_parameter_coords), 
                                            total=num_prior_values):      
                                                            
            prior_matrix = np.squeeze(
                prior.construct_prior_array(
                    spectral_parameters = {param_key: hyperparametervalues[param_idx] for param_idx, param_key in enumerate(prior_spectral_params.keys())},
                    spatial_parameters = {param_key: hyperparametervalues[num_spec_params+ param_idx] for param_idx, param_key in enumerate(prior_spatial_params.keys())},
                    normalise=True)
                    )
            
            nans+=np.sum(np.isnan(prior_matrix))
            prior_matrices[_inner_idx,...] = prior_matrix

        return prior_matrices, nans

        

    

    


    def nuisance_log_marginalisation(self, measured_event_data: EventData) -> list:

        prior_marged_shapes, _ = self.prior_gen(Nevents=measured_event_data.Nevents)

        marg_partial = functools.partial(
            self.observation_nuisance_marg, 
            prior_matrix_list=self.prior_matrix_list)

        with Pool(self.numcores) as pool:
            marg_results = pool.map(marg_partial, measured_event_data.data)

        marg_results = np.asarray(marg_results)

        reshaped_marg_results = []
        for _prior_idx in range(len(self.priors)):
            nice_marg_results = np.squeeze(np.vstack(marg_results[:,_prior_idx]))
            logging.info('\nnice shape: ', nice_marg_results.shape)
            nice_marg_results = nice_marg_results.reshape(prior_marged_shapes[_prior_idx])
            reshaped_marg_results.append(nice_marg_results)

        return reshaped_marg_results
    

    def add_log_nuisance_marg_results(self, new_log_marg_results: np.ndarray) -> None:
        """Add log nuisance marginalisation results to those within the class.

        Args:
            new_log_marg_results (np.ndarray): The log likelihood values after 
                marginalising over nuisance parameters.
        """
        self.log_margresults = np.append(self.log_margresults, new_log_marg_results, axis=0)
    
    def apply_direchlet_stick_breaking_direct(self, 
                                              xi_axes: list | tuple, 
                                              depth: int) -> np.ndarray | float:
        direchletmesh = 1

        for _dirichlet_i in range(depth):
            direchletmesh*=(1-xi_axes[_dirichlet_i])
        
        not_max_depth = depth!=len(xi_axes)
        if not_max_depth:
            direchletmesh*=xi_axes[depth]

        return direchletmesh
    
    def _mixture_input_filtering(self, 
                                 mixture_param_set: list | dict | ParameterSet = None, 
                                 log_margresults: list | tuple | np.ndarray = None):
        
        if mixture_param_set is None:
            mixture_param_set = self.mixture_param_set
            if self.mixture_param_set is None:
                raise Exception("Mixture axes not specified")
        else:
            self.mixture_param_set = ParameterSet(mixture_param_set)

        if not(self.priors is None):
            if len(mixture_param_set)!=len(self.priors)-1:
                raise Exception(f""""Number of mixture axes does not match number of components (minus 1). Please check your inputs.
Number of mixture axes is {len(mixture_param_set)}
and number of prior components is {len(self.priors)}.""")
            
        if log_margresults is None:
            log_margresults = self.log_margresults

        return mixture_param_set, log_margresults
    
    def create_mixture_comp(self, 
                            prior_idx: int, 
                            log_margresults_for_idx:np.ndarray,
                            mix_axes_mesh:list[np.ndarray],
                            final_output_shape:list,
                            prior_axes,
                            hyper_idx):
        # Including 'event' and mixture axes for eventual __non__ expansion into
        prior_axis_instance = list(range(1+len(mix_axes_mesh)))

        for length_of_axis in log_margresults_for_idx.shape[1:]:

            final_output_shape.append(length_of_axis)

            prior_axis_instance.append(hyper_idx)

            hyper_idx+=1

        prior_axes.append(prior_axis_instance)


        mixcomp = np.expand_dims(np.log(
            self.apply_direchlet_stick_breaking_direct(xi_axes=mix_axes_mesh, depth=prior_idx)), 
            axis=(*np.delete(np.arange(log_margresults_for_idx.ndim+len(mix_axes_mesh)), 
                                np.arange(len(mix_axes_mesh))+1),)) 


        mixcomp=mixcomp+np.expand_dims(log_margresults_for_idx, 
                                    axis=(*(np.arange(len(mix_axes_mesh))+1),)
                                )
        return mixcomp, hyper_idx
        
    
            
    def create_discrete_mixture_log_hyper_likelihood(self, 
                                                     log_margresults: list | tuple | np.ndarray = None,
                                                     mixture_param_set: list | ParameterSet = None, 
                                                     ):
        
        mixture_param_set, log_margresults = self._mixture_input_filtering(mixture_param_set, log_margresults)

        mix_axes_mesh = np.meshgrid(*mixture_param_set.axes, indexing='ij')

        # To get around the fact that every component of the mixture can be a different shape
        final_output_shape = [len(log_margresults), *np.arange(len(mixture_param_set)), ]

        # Axes for each of the priors to __not__ expand in
        prior_axes = []

        # Counter for how many hyperparameter axes we have used
        hyper_idx = len(mixture_param_set)+1


        # Creating components of mixture for each prior
        mixture_array_comp_list = []
        for prior_idx, log_margresults_for_idx in enumerate(log_margresults):

            mix_comp, hyper_idx  = self.create_mixture_comp( 
                            prior_idx = prior_idx, 
                            log_margresults_for_idx = log_margresults_for_idx,
                            mix_axes_mesh=mix_axes_mesh,
                            final_output_shape=final_output_shape,
                            prior_axes=prior_axes,
                            hyper_idx=hyper_idx)
            
            mixture_array_comp_list.append(mix_comp)
            
        # Making all the mixture components the same shape that is compatible for adding together (in logspace)
        for _prior_idx, mixture_array in enumerate(mixture_array_comp_list):
            axis = []
            for _axis_idx in range(len(final_output_shape)):
                if not(_axis_idx in prior_axes[_prior_idx]):
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

        if integrator is None:
            integrator = self.logspace_integrator
        if log_hyper_priormesh is None:
            hyper_val_list = []
            log_prior_val_list = []
            


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
        
        packed_data['parameter_specifications']                 = self.parameter_specifications
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
        except Exception as excpt:
            warnings.warn(f"""Something went wrong when trying to save to the specified directory. 
Saving to current working directory as hyper_class_data.yaml: {excpt}""")
            save_config_file("hyper_class_data.yaml", file_name)
