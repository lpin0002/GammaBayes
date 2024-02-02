import numpy as np
import functools
from tqdm import tqdm
from gammabayes.utils import (
    iterate_logspace_integration, 
    logspace_riemann, 
    update_with_defaults,
    apply_direchlet_stick_breaking_direct
)

from gammabayes.utils.config_utils import save_config_file
from gammabayes import EventData, Parameter, ParameterSet
from gammabayes.priors import discrete_logprior
from multiprocessing.pool import ThreadPool as Pool
import os, warnings, logging, time, h5py

class discrete_brute_scan_hyperparameter_likelihood(object):
    def __init__(self, log_priors: list[discrete_logprior] | tuple[discrete_logprior] = None, 
                 log_likelihood: callable = None, 
                 axes: list[np.ndarray] | tuple[np.ndarray] | None=None,
                 nuisance_axes: list[np.ndarray] | tuple[np.ndarray] | None = None,
                 parameter_specifications: dict | list[ParameterSet] | dict[ParameterSet] = {}, 
                 log_likelihoodnormalisation: np.ndarray | float = 0., 
                 log_margresults: np.ndarray | None = None, 
                 mixture_param_specifications: list[np.ndarray] | tuple[np.ndarray] | None = None,
                 log_hyperparameter_likelihood: np.ndarray | float = 0., 
                 log_posterior: np.ndarray | float = 0., 
                 iterative_logspace_integrator: callable = iterate_logspace_integration,
                 logspace_integrator: callable = logspace_riemann,
                 log_prior_matrix_list: list[np.ndarray] | tuple[np.ndarray] = None,
                 log_marginalisation_regularisation: float  = None,
                 no_priors_on_init: bool = False,
                 no_likelihood_on_init: bool = False
                 ):
        """
        Initializes a discrete brute scan hyperparameter likelihood object.

        Args:
            log_priors (list[discrete_logprior] | tuple[discrete_logprior], optional): 
                Priors for the log probabilities of discrete input values. Defaults to None.
            
            log_likelihood (callable, optional): A callable object to compute 
                the log likelihood. Defaults to None.
            
            axes (list[np.ndarray] | tuple[np.ndarray] | None, optional): Axes 
                that the measured event data can take. Defaults to None.
            
            nuisance_axes (list[np.ndarray] | tuple[np.ndarray] | None, optional): 
                Axes that the nuisance parameters can take. Defaults to None.
            
            parameter_specifications (dict, optional): Specifications for 
                parameters involved in the likelihood estimation. Defaults to an empty dictionary.
            
            log_likelihoodnormalisation (np.ndarray | float, optional): 
                Normalization for the log likelihood. Defaults to 0.
            
            log_margresults (np.ndarray | None, optional): Results of 
                marginalization, expressed in log form. Defaults to None.
            
            mixture_param_specifications (list[np.ndarray] | tuple[np.ndarray] | None, optional): 
                Specifications for the mixture fraction parameters. Defaults to None.
            
            log_hyperparameter_likelihood (np.ndarray | float, optional): 
                Log likelihood of hyperparameters (penultimate result). Defaults to 0.
            
            log_posterior (np.ndarray | float, optional): Log of the 
                hyperparameter posterior probability (final result of class). 
                Defaults to 0.
            
            iterative_logspace_integrator (callable, optional): A callable 
                function for iterative integration in log space. 
                Defaults to iterate_logspace_integration.
            
            logspace_integrator (callable, optional): A callable function for 
                performing single dimension integration in log space. 
                Defaults to logspace_riemann.
            
            log_prior_matrix_list (list[np.ndarray] | tuple[np.ndarray], optional): 
                List or tuple of matrices representing the log of the prior matrices
                for the given hyperparameters over the nuisance parameter axes. 
                Defaults to None.
            
            log_marginalisation_regularisation (float, optional): Regularisation 
                parameter for log marginalisation, should not change normalisation
                of final result but will minimise effects of numerical instability. Defaults to None.
        """

        
        self.log_priors             = log_priors

        self.log_likelihood         = log_likelihood

        # Making it so that it has to be an active choice not to include priors and likelihoods
        self.no_priors_on_init      =False,
        self.no_likelihood_on_init  =False

        self.no_priors_on_init = no_priors_on_init

        self.no_likelihood_on_init = no_likelihood_on_init


        # Axes for the __reconstructed__ values
        self.axes               = axes

        # Axes for the "true" values
        self.nuisance_axes     = self._handle_nuisance_axes(nuisance_axes)


        self._handle_parameter_specification(parameter_specifications=parameter_specifications)

        # Currently required as the normalisation of the IRFs isn't natively consistent
        self.log_likelihoodnormalisation            = np.asarray(log_likelihoodnormalisation)

        # Log-space integrator for multiple dimensions (kind of inefficient at the moment)
        self.iterative_logspace_integrator      = iterative_logspace_integrator

        # Single dimension log-space integrator (fully vectorised)
        self.logspace_integrator            = logspace_integrator

        # Doesn't have to be initialised here, but you can do it if you want
        self.mixture_param_specifications          = ParameterSet(mixture_param_specifications)

        # Used as regularisation to avoid
        self.log_marginalisation_regularisation = log_marginalisation_regularisation

        # From here it's the initialisation of results, but the arguments should be able
            # to be used as initialisation if you have produced you results in another
            # way, assuming that they match the expected inputs/outputs of what is
            # used/produce here.

        self.log_margresults                    = log_margresults
        self.log_hyperparameter_likelihood      = log_hyperparameter_likelihood
        self.log_posterior                      = log_posterior
        self.log_prior_matrix_list              = log_prior_matrix_list


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
                    return self.log_priors[0].axes
                except AttributeError:
                    raise Exception("Dependent value axes used for calculations not given.")
        return nuisance_axes


    def _handle_parameter_specification(self, 
                                        parameter_specifications: dict | ParameterSet):
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

        if _num_parameter_specifications>0:

            if type(parameter_specifications)==dict:

                for single_prior_parameter_specifications in parameter_specifications.items():

                    parameter_set = ParameterSet(single_prior_parameter_specifications)

                    formatted_parameter_specifications.append(parameter_set)

            elif type(parameter_specifications)==list:
                formatted_parameter_specifications = parameter_specifications

        try:
            self._num_priors = len(self.log_priors)
        except TypeError as excpt:
            logging.warning(f"An error occured when trying to calculate the number of priors: {excpt}")
            self._num_priors = _num_parameter_specifications

        if not self.no_priors_on_init or (self.log_priors is not None):

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
        



    # Most import method in this whole class. If debugging be firmly aware of
        # - prior matrices normalisations
        # - log_likelihood normalisations. 
            # Must be normalised over __measured/reconstructed__ axes __not__
            # over the dependent axes
        # - numerical stability
            # Tried my best to avoid this, but if input values are close to
            # numerical precision, watch out
        # - Event Values
            # Garbage In - Garbage Out.
            # If you are simulating your events in particular, make sure that
            # you can plot your produced events for a prior and match the shape
            # near exactly for a high number of events to the functional form 
            # of the prior directly (going to have to scale one or the other,
            # shape is what matters)
    def observation_nuisance_marg(self, 
                                  event_vals: list | np.ndarray, 
                                  log_prior_matrix_list: list[np.ndarray] | tuple[np.ndarray]) -> np.ndarray:
        """
        Calculates the marginal log likelihoods for observations by integrating over nuisance parameters. 
        This method creates a meshgrid of event values and nuisance axes, computes the log likelihood values, 
        and integrates these values with the log prior matrices.

        Args:
            event_vals (list | np.ndarray): The event values for which the marginal log likelihoods are to be 
                                            computed. Can be a list or a numpy ndarray.
            log_prior_matrix_list (list[np.ndarray] | tuple[np.ndarray]): A list or tuple of numpy ndarrays 
                                                                        representing log prior matrices.

        Returns:
            np.ndarray: An array of marginal log likelihood values corresponding to each set of event values 
                        and each log prior matrix.

        Notes:
            - This method uses meshgrid to create a grid of event values and nuisance axes.
            - It adjusts the log likelihood values by subtracting the normalization factor.
            - The integration over nuisance parameters is performed using the specified iterative logspace integrator.
        """


        meshvalues  = np.meshgrid(*event_vals, *self.nuisance_axes, indexing='ij')
        flattened_meshvalues = [meshmatrix.flatten() for meshmatrix in meshvalues]
        
        log_likelihoodvalues = np.squeeze(self.log_likelihood(*flattened_meshvalues).reshape(meshvalues[0].shape))

        log_likelihoodvalues = log_likelihoodvalues - self.log_likelihoodnormalisation
        
        all_log_marg_results = []

        for log_prior_matrices in log_prior_matrix_list:
            # Transpose is because of the convention for where I place the 
                # axes of the true values (i.e. first three indices of the prior matrices)
                # and numpy doesn't support this kind of matrix addition nicely
            logintegrandvalues = (np.squeeze(log_prior_matrices).T+np.squeeze(log_likelihoodvalues).T).T
            
            single_parameter_log_margvals = self.iterative_logspace_integrator(logintegrandvalues,   
                axes=self.nuisance_axes)

            all_log_marg_results.append(np.squeeze(single_parameter_log_margvals))

        
        return np.array(all_log_marg_results, dtype=object)


    def prior_gen(self, Nevents: int) -> tuple[(np.ndarray, list)]:
        """
        Generates and returns the shapes and matrices of priors for a given number of events. This method handles 
        the generation of prior matrices, taking into account both efficient and inefficient construction methods, 
        and tracks NaN values that may arise during this process.

        Args:
            Nevents (int): The number of events to be considered in the generation of prior matrices.

        Returns:
            list: A tuple containing two elements:
                1. An array of shapes of the non-flattened prior matrices.
                2. The list of prior matrices generated.

        Notes:
            - Sets numpy error handling to ignore warnings and errors arising from division by zero or invalid operations.
            - Initializes `log_prior_matrix_list` if it does not exist and populates it by iterating over `log_priors`.
            - Differentiates between efficient and inefficient methods of constructing prior matrices.
            - Reshapes the prior matrices to minimize memory usage and enhance computation efficiency.
            - Tracks the number of NaN values encountered in the prior matrices.
            - Provides debugging advice related to the handling of prior matrices, especially in the context of 
            multiprocessing environments.
        """


        # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
        np.seterr(divide='ignore', invalid='ignore')


        # Array that contains the shapes of the non-flattened shapes of the prior matrices
        prior_marged_shapes = np.empty(shape=(len(self.log_priors),), dtype=object)

        # Keeps track of the nan values that pop up in the prior matrices
            # Generally not great. Good thing to check when debugging.
        nans =  0

        if self.log_prior_matrix_list is None:
            logging.info("log_prior_matrix_list does not exist. Constructing priors.")
            log_prior_matrix_list =     []

            for _prior_idx, log_prior in tqdm(enumerate(self.log_priors), 
                                   total=len(self.log_priors), 
                                   desc='Setting up prior matrices'):
                

                prior_parameter_specifications = self.parameter_specifications[_prior_idx].scan_format

                prior_spectral_params   = prior_parameter_specifications['spectral_parameters']
                prior_spatial_params    = prior_parameter_specifications['spatial_parameters']


                prior_marged_shapes[_prior_idx] = (Nevents, 
                                                   *[parameter_specification.size for parameter_specification in prior_spectral_params.values()],
                                                   *[parameter_specification.size for parameter_specification in prior_spatial_params.values()])


                if log_prior.efficient_exist:
                    log_prior_matrices = np.squeeze(
                        log_prior.construct_prior_array(
                            spectral_parameters = prior_spectral_params,
                            spatial_parameters =  prior_spatial_params,
                            normalise=True)
                            )
                    
                    # Computation is more efficient/faster of the arrays 
                        # are as flattened as possible. Also minimises memory
                    log_prior_matrices = log_prior_matrices.reshape(*self.log_likelihoodnormalisation.shape, -1)
                    
                    # Each index of the matrices that is nan shows up as 1, otherwise 0
                        # To figure out how many nans there are, you can just add the 1's
                    nans+=np.sum(np.isnan(log_prior_matrices))

                else:

                    # If the prior does not have the mesh inefficient function, that allows
                        # Easy construction of the prior matrices, then each matrix for each 
                        # combination of the hyperparameter values is calculated one-by-one
                    log_prior_matrices, nans = self._mesh_inefficient_prior_construction( 
                                             log_prior=log_prior, prior_idx=_prior_idx,
                                             prior_spectral_params=prior_spectral_params, 
                                             prior_spatial_params=prior_spatial_params, 
                                             Nevents=Nevents, prior_marged_shapes=prior_marged_shapes,
                                             nans=nans
                                             )

                # Making sure the output is a numpy array
                log_prior_matrices = np.asarray(log_prior_matrices, dtype=float)

                log_prior_matrix_list.append(log_prior_matrices)

            logging.debug(f"Total cumulative number of nan values within all prior matrices: {nans}")
            

            # If 
            self.log_prior_matrix_list = log_prior_matrix_list
            logging.debug(f"""If debugging, check how you are treating your 
prior matrices. Generally they are saved as an attribute of the class, but this 
can be problematic during multiprocessing, so I would advise using the output 
of this function for the subsequent steps directly. Otherwise, initialise the 
class before the multiprocessing or make sure that it isn't part of the actual 
'multi' in the processing.""")


            return prior_marged_shapes, log_prior_matrix_list
        

    def _mesh_inefficient_prior_construction(self, 
                                             log_prior: discrete_logprior, 
                                             prior_idx: int,
                                             prior_spectral_params: dict, 
                                             prior_spatial_params: dict, 
                                             Nevents: int, 
                                             prior_marged_shapes: list[np.ndarray] | list[list] | list[tuple],
                                             nans: int
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
            num_prior_values =  1
            parameter_matrix_shape = ()
            flattened_hyper_parameter_coords = [[]]


        log_prior_matrices  = np.empty(shape = (
            num_prior_values,
            *np.squeeze(self.log_likelihoodnormalisation).shape,
            )
            )


        prior_marged_shapes[prior_idx] = (Nevents, *parameter_matrix_shape)



        for _inner_idx, hyperparametervalues in enumerate(flattened_hyper_parameter_coords):

            prior_matrix = np.squeeze(
                log_prior.construct_prior_array(
                    spectral_parameters = {
                        param_key: hyperparametervalues[param_idx] for param_idx, param_key in enumerate(prior_spectral_params.keys())
                        },
                    spatial_parameters = {
                        param_key: hyperparametervalues[num_spec_params+ param_idx] for param_idx, param_key in enumerate(prior_spatial_params.keys())
                        },
                    normalise=True)
                    )
            
            nans +=     np.sum(np.isnan(prior_matrix))
            log_prior_matrices[_inner_idx,...] =    prior_matrix

        return log_prior_matrices, nans

        


    def nuisance_log_marginalisation(self, measured_event_data: EventData) -> list:
        """
        Performs log marginalisation over nuisance parameters for measured event data. This method generates 
        marginalisation results for each event in the measured data and then reshapes these results to align with 
        the prior shapes. It also calculates the log marginalisation regularisation value.

        Args:
            measured_event_data (EventData): The measured event data containing information such as the number of 
                                            events and event-specific data.

        Returns:
            list: A list of reshaped marginalisation results, each corresponding to a different prior.

        Notes:
            - The method first generates prior shapes and matrices using the `prior_gen` method.
            - It calculates marginalisation results for each event in the measured data using `observation_nuisance_marg`.
            - The results are reshaped to make the output iterable around the priors, which is more convenient for further processing.
            - Determines the log marginalisation minimum and maximum values for each reshaped result, excluding 
            `-np.inf` values, to calculate the regularisation term.
            - Sets `log_marginalisation_regularisation` to the mean difference between log marginalisation maximums and minimums.
        """

        prior_marged_shapes, _ = self.prior_gen(Nevents=measured_event_data.Nevents)

        marg_results = [self.observation_nuisance_marg(
            log_prior_matrix_list=self.log_prior_matrix_list, 
            event_vals=event_data) for event_data in measured_event_data.data]
                
        marg_results = np.asarray(marg_results)


        # Making the output iterable around the priors, not the events.
            # Much easier to work with.
        reshaped_marg_results = []
        for _prior_idx in range(len(self.log_priors)):
            stacked_marg_results = np.squeeze(np.vstack(marg_results[:,_prior_idx]))

            logging.info('\nstacked_marg_results shape: ', stacked_marg_results.shape)

            stacked_marg_results = stacked_marg_results.reshape(prior_marged_shapes[_prior_idx])

            reshaped_marg_results.append(stacked_marg_results)


        log_marg_mins = np.asarray([np.nanmin(reshaped_marg_result[reshaped_marg_result != -np.inf]) for reshaped_marg_result in reshaped_marg_results])
        log_marg_maxs = np.asarray([np.nanmax(reshaped_marg_result[reshaped_marg_result != -np.inf]) for reshaped_marg_result in reshaped_marg_results])

        # Adaptively set the regularisation based on the range of values in the
        # log marginalisation results. Trying to keep them away from ~-600 or ~600
        # generally precision goes down to ~1e-300 (base 10)
        self.log_marginalisation_regularisation = np.abs(0.3*np.mean(np.diff(log_marg_maxs-log_marg_mins)))


        return reshaped_marg_results
    

    def add_log_nuisance_marg_results(self, new_log_marg_results: np.ndarray):
        """
        Extends each array in the existing log marginalisation results with corresponding new log marginalisation results. 
        This method iterates over each array in `self.log_margresults` and the new results, extending each existing array 
        with the corresponding new results. 

        Args:
            new_log_marg_results (np.ndarray): An array containing new log marginalisation results. Each element in this 
                                            array corresponds to and will be appended to the respective array in 
                                            `self.log_margresults`.

        Notes:
            - It's assumed that `self.log_margresults` is a list of lists (or arrays) where each sublist corresponds 
            to a set of log marginalisation results.
            - `new_log_marg_results` should have the same length/number of priors as `self.log_margresults` to ensure 
            correct pairing and extension of each sublist.
            - This method uses list comprehension to iterate and extend each corresponding sublist.
        """
        self.log_margresults = [log_margresult.extend(new_log_marg_result) for log_margresult, new_log_marg_result in zip(self.log_margresults, new_log_marg_results)]

    
    def _mixture_input_filtering(self, 
                                 mixture_param_specifications: list | dict | ParameterSet = None, 
                                 log_margresults: list | tuple | np.ndarray = None):
        """
        Filters and validates the input for mixture parameters and log marginalisation results. 
        This method ensures that the provided mixture parameters and log marginalisation results 
        are consistent with the expected format and the current state of the object.

        Args:
            mixture_param_specifications (list | dict | ParameterSet, optional): The set of parameters for 
                                                                    the mixture model. Can be a list, 
                                                                    a dictionary, or a ParameterSet object. 
                                                                    Defaults to None.
            log_margresults (list | tuple | np.ndarray, optional): Log marginalisation results. 
                                                                Can be a list, tuple, or numpy ndarray. 
                                                                Defaults to None.

        Returns:
            tuple: A tuple containing the validated mixture_param_specifications and log_margresults.

        Raises:
            Exception: Raised if mixture_param_specifications is not specified or if the number of mixture 
                    axes does not match the number of components (minus 1) in the log priors.

        Notes:
            - If `mixture_param_specifications` is not provided, it defaults to `self.mixture_param_specifications`.
            - An exception is raised if no mixture axes are specified.
            - If `log_margresults` is not provided, it defaults to `self.log_margresults`.
            - The method checks the consistency between the number of mixture axes and the number 
            of prior components.
        """

        
        if mixture_param_specifications is None:
            mixture_param_specifications = self.mixture_param_specifications
            if self.mixture_param_specifications is None:
                raise Exception("Mixture axes not specified")
        else:
            self.mixture_param_specifications = ParameterSet(mixture_param_specifications)

        if not(self.log_priors is None):
            if len(mixture_param_specifications)!=len(self.log_priors)-1:
                raise Exception(f""""Number of mixture axes does not match number of components (minus 1). Please check your inputs.
Number of mixture axes is {len(mixture_param_specifications)}
and number of prior components is {len(self.log_priors)}.""")
            
        if log_margresults is None:
            log_margresults = self.log_margresults

        return mixture_param_specifications, log_margresults
    
    def create_mixture_comp(self, 
                            prior_idx: int, 
                            log_margresults_for_idx:np.ndarray,
                            mix_axes_mesh:list[np.ndarray],
                            final_output_shape:list,
                            prior_axes_indices: list[list],
                            hyper_idx: int):
        """
        Creates a single component (i.e. the component for __a__ prior) of the mixture model. 
        This method combines log marginalisation results for a given prior index with the 
        mixture axis information to form a component of the mixture model. 

        Args:
            prior_idx (int): The index of the prior for which the mixture component is being created.

            log_margresults_for_idx (np.ndarray): The log marginalisation results corresponding to the given prior index.

            mix_axes_mesh (list[np.ndarray]): A list of numpy arrays representing the meshgrid of mixture axes.

            final_output_shape (list): A list to store the final shape of the output mixture component.

            prior_axes_indices (list[list]): A list to keep track of the indices in each prior for the final output.

            hyper_idx (int): The current index in the hyperparameter space.

        Returns:
            tuple: A tuple containing the created mixture component (numpy array) and the updated hyperparameter index (int).

        Notes:
            - The method calculates the log mixture component using the Dirichlet stick breaking process.
            - It expands and combines the calculated mixture component with the log marginalisation results.
            - Updates `final_output_shape` and `prior_axes_indices` to reflect the new dimensions and indices after 
            combining the mixture component with the log marginalisation results.
            - The method returns the new mixture component and the updated hyperparameter index.
        """
        # Including 'event' and mixture axes for eventual __non__ expansion into
        single_prior_axes_indices_instance = list(range(1+len(mix_axes_mesh)))

        # First index of 'log_margresults_for_idx' should just be the number of events
        for length_of_axis in log_margresults_for_idx.shape[1:]:

            final_output_shape.append(length_of_axis)

            single_prior_axes_indices_instance.append(hyper_idx)

            # Keeping track of the indices in each prior for the final output
            hyper_idx+=1

        prior_axes_indices.append(single_prior_axes_indices_instance)


        mixcomp = np.expand_dims(np.log(
            apply_direchlet_stick_breaking_direct(mixtures_fractions=mix_axes_mesh, depth=prior_idx)), 
            axis=(*np.delete(np.arange(log_margresults_for_idx.ndim+len(mix_axes_mesh)), 
                                np.arange(len(mix_axes_mesh))+1),)) 


        mixcomp=mixcomp+np.expand_dims(log_margresults_for_idx, 
                                    axis=(*(np.arange(len(mix_axes_mesh))+1),)
                                )
        return mixcomp, hyper_idx
        
    
            
    def create_discrete_mixture_log_hyper_likelihood(self, 
                                                     log_margresults: list | tuple | np.ndarray = None,
                                                     mixture_parameter_set: list | ParameterSet = None, 
                                                     ) -> np.ndarray:
        """
        Constructs the logarithmic hyperparameter likelihood of a discrete mixture model. This method processes log 
        marginalisation results and mixture parameter sets to compute the log likelihood of hyperparameters in a 
        discrete mixture model.

        Args:
            log_margresults (list | tuple | np.ndarray, optional): Log marginalisation results for each prior. 
                                                                Defaults to None.
            mixture_parameter_set (list | ParameterSet, optional): Parameters for the mixture model, either as a 
                                                                    list or a ParameterSet object. Defaults to None.

        Returns:
            np.ndarray: The logarithmic likelihood of hyperparameters for the discrete mixture model.

        Notes:
            - Regularises log marginalisation results by adding `log_marginalisation_regularisation`.
            - Filters and validates the mixture parameter set and log marginalisation results using `_mixture_input_filtering`.
            - Creates a meshgrid for the mixture axes.
            - Initializes the final output shape and prior axes indices for the mixture components.
            - Iteratively constructs mixture components for each prior using `create_mixture_comp`.
            - Ensures all mixture components have compatible shapes for combination.
            - Combines all mixture components into a single array to represent the combined mixture.
            - Calculates the log hyperparameter likelihood by summing over the combined mixture components.
            - Adjusts the final likelihood by undoing the regularisation applied initially.
        """
        logging.debug(f"log_marginalisation_regularisation: {self.log_marginalisation_regularisation}")
        log_margresults = [log_margresult + self.log_marginalisation_regularisation for log_margresult in log_margresults]
        
        mixture_parameter_set, log_margresults = self._mixture_input_filtering(mixture_parameter_set, log_margresults)

        mix_axes_mesh = np.meshgrid(*mixture_parameter_set.axes, indexing='ij')

        # To get around the fact that every component of the mixture can be a different shape
        final_output_shape = [len(log_margresults), *np.arange(len(mixture_parameter_set)), ]

        # Axes for each of the priors to __not__ expand in
        prior_axes_indices = []

        # Counter for how many hyperparameter axes we have used
        hyper_idx = len(mixture_parameter_set)+1


        # Creating components of mixture for each prior
        mixture_array_comp_list = []
        for prior_idx, log_margresults_for_idx in enumerate(log_margresults):

            mix_comp, hyper_idx  = self.create_mixture_comp( 
                            prior_idx = prior_idx, 
                            log_margresults_for_idx = log_margresults_for_idx,
                            mix_axes_mesh=mix_axes_mesh,
                            final_output_shape=final_output_shape,
                            prior_axes_indices=prior_axes_indices,
                            hyper_idx=hyper_idx)
            
            mixture_array_comp_list.append(mix_comp)
            
        # Making all the mixture components the same shape that is compatible for adding together (in logspace)
        for _prior_idx, mixture_array in enumerate(mixture_array_comp_list):
            axis = []
            for _axis_idx in range(len(final_output_shape)):
                if not(_axis_idx in prior_axes_indices[_prior_idx]):
                    axis.append(_axis_idx)
            axis = tuple(axis)

            mixture_array   = np.expand_dims(mixture_array, axis=axis)

            mixture_array_comp_list[_prior_idx] = mixture_array

        # Now to combine the results for all the data and to 
        combined_mixture = -np.inf
        for mixture_component in mixture_array_comp_list:
            mixture_component = mixture_component - self.log_marginalisation_regularisation
            combined_mixture = np.logaddexp(combined_mixture, mixture_component)


        # Multiplying the likelihoods of all the events together
        log_hyperparameter_likelihood = np.sum(combined_mixture, axis=0)

        # Undoing the regularisation for proper output
        log_hyperparameter_likelihood = log_hyperparameter_likelihood - self.log_marginalisation_regularisation

        return log_hyperparameter_likelihood
    

    def split_into_batches(self, 
                           log_margresults: list, 
                           num_in_batches: int) -> list:
        """
        Splits the log marginalisation results into batches. This method divides the provided log marginalisation 
        results into smaller batches, each containing a specified number of elements.

        Args:
            log_margresults (list): A list containing log marginalisation results. Each element in the list 
                                    corresponds to a different prior and is assumed to be an array.
            num_in_batches (int): The number of elements to include in each batch.

        Returns:
            list: A list of batched log marginalisation results. Each batch is a list of arrays, one for each 
                prior, with each array containing a subset of the original log marginalisation results.

        Notes:
            - The method iterates over the log marginalisation results, creating batches that contain a segment 
            of the results from each prior.
            - Each batch includes results from the same index range across all priors.
            - The size of each batch is determined by `num_in_batches`, with the last batch possibly being smaller 
            if the total number of results is not a multiple of `num_in_batches`.
        """
        batched_log_margresults = []
        for batch_idx in range(0, len(log_margresults[0]), num_in_batches):
            batched_log_margresults.append(
                [single_prior_log_margresults[batch_idx:batch_idx+num_in_batches, ...] for single_prior_log_margresults in log_margresults])
            
        return batched_log_margresults


    
    # The above 'create_discrete_mixture_log_hyper_likelihood' method is 
        # vectorised. This means computation is pretty well optimised, 
        # but can lead to memory problems as the entire matrix of shape
            # (Nevents, 
            # *[mixture_parameters], 
            # *[all spectral parameters], 
            # *[all spatial parameters])
        # has to be loaded in all at once. So the 'batch processing' method
        # was implemented to reduce the number of elements in the first dimension
        # (i.e. the number of events) which is what 'num_in_mixture_batch' is 
        # referring to
    def batch_process_create_discrete_mixture(self, 
                                              log_margresults,
                                              mixture_parameter_set, 
                                              num_in_mixture_batch: int = 10) -> np.ndarray:
        """
        Processes the creation of a discrete mixture model in batches. This method is designed to handle memory 
        constraints by dividing the computation into smaller batches, each containing a subset of event data.

        Args:
            log_margresults: The log marginalisation results to be processed. These results are used to create 
                            the discrete mixture model.
            mixture_parameter_set: The set of parameters for the mixture model. This set defines the axes over 
                                which the mixture model is computed.
            num_in_mixture_batch (int, optional): The number of events to include in each batch. Defaults to 10.

        Returns:
            np.ndarray: The logarithmic hyperparameter likelihood for the entire dataset, obtained by summing 
                        over the log results of the batches.

        Notes:
            - The method uses `split_into_batches` to divide the log marginalisation results into smaller batches 
            based on `num_in_mixture_batch`.
            - It then processes each batch independently using the `create_discrete_mixture_log_hyper_likelihood` method.
            - The results from all batches are summed to get the final log hyperparameter likelihood.
            - This batching approach helps manage memory usage by reducing the number of events processed at one time.
        """
        
        batched_log_margresults = self.split_into_batches(log_margresults=log_margresults, 
                                                          num_in_batches=num_in_mixture_batch)
        

        log_hyperparameter_likelihood_batches = [self.create_discrete_mixture_log_hyper_likelihood(
            log_margresults=batch_of_log_margresults,
            mixture_parameter_set=mixture_parameter_set,
        ) for batch_of_log_margresults in batched_log_margresults]

        log_hyperparameter_likelihood = np.sum(log_hyperparameter_likelihood_batches, axis=0)

        return log_hyperparameter_likelihood
        
        
    
    def update_hyperparameter_likelihood(self, 
                                           log_hyperparameter_likelihood: np.ndarray
                                           ) -> np.ndarray:
        """
        Updates the existing log hyperparameter likelihood with new values. This method adds the provided 
        log hyperparameter likelihood values to the current values stored in the object.

        Args:
            log_hyperparameter_likelihood (np.ndarray): An array of new log hyperparameter likelihood values to be added 
                                                        to the existing log hyperparameter likelihood.

        Returns:
            np.ndarray: The updated log hyperparameter likelihood after adding the new values.

        Notes:
            - This method is used to cumulatively update the log hyperparameter likelihood.
            - The method assumes that `self.log_hyperparameter_likelihood` has been initialized and is in a compatible 
            format with the `log_hyperparameter_likelihood` argument.
        """
        self.log_hyperparameter_likelihood += log_hyperparameter_likelihood

        return self.log_hyperparameter_likelihood
    

    # TODO: Make this usable
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

    # Private method for classes that inherit this classes behaviour åto use
        # and don't have to re-write many of the same lines again
    def _pack_data(self, reduce_mem_consumption: bool = True, h5f=None, file_name='temporary_file.h5') -> dict:
        if h5f is None:
            h5f = h5py.File(file_name, 'w')  # Replace 'temporary_file.h5' with desired file name
        
        
        h5f.attrs['log_marginalisation_regularisation'] = self.log_marginalisation_regularisation
        if self.log_likelihoodnormalisation is not None:

            h5f.create_dataset('log_likelihoodnormalisation', data=np.asarray(self.log_likelihoodnormalisation, dtype=np.float64))

        
        # Handling numpy arrays
        if self.axes is not None:

            axes_group = h5f.create_group('axes') 
            for axis_idx, axis in enumerate(self.axes):
                axis_dataset = axes_group.create_dataset(f"{axis_idx}", data=axis)


                
        if self.nuisance_axes is not None:
            nuisance_axes_group = h5f.create_group('nuisance_axes') 
            for nuisance_axis_idx, nuisance_axis in enumerate(self.nuisance_axes):
                nuisance_axis_dataset = nuisance_axes_group.create_dataset(f"{nuisance_axis_idx}", data=nuisance_axis)


        if self.log_margresults is not None:
            h5f.create_dataset('log_margresults', data=self.log_margresults)

        if self.log_hyperparameter_likelihood is not None:
            h5f.create_dataset('log_hyperparameter_likelihood', data=self.log_hyperparameter_likelihood)


        if self.log_posterior is not None:
            h5f.create_dataset('log_posterior', data=self.log_posterior)


        # Packing ParameterSet objects
        if self.parameter_specifications is not None:
            prior_param_set_group = h5f.create_group('prior_param_set')

            for prior_idx, (prior, single_prior_param_set) in enumerate(zip(self.log_priors, self.parameter_specifications)):
                if type(single_prior_param_set) == ParameterSet:
                    single_prior_param_group = prior_param_set_group.create_group(prior.name)
                    single_prior_param_group = single_prior_param_set.pack(h5f=single_prior_param_group)

        if self.mixture_param_specifications is not None:

            mixture_param_specifications_group = h5f.create_group('mixture_param_specifications')

            mixture_param_specifications_group = self.mixture_param_specifications.pack(h5f=mixture_param_specifications_group)


        if not reduce_mem_consumption:

            if self.log_prior_matrix_list is not None:
                log_prior_matrix_list_group = h5f.create_group('log_prior_matrix_list')

                for prior, single_prior_log_matrix_list in zip(self.log_priors, self.log_prior_matrix_list):
                    log_prior_matrix_list_group.create_dataset(prior.name, single_prior_log_matrix_list)


        return h5f
    

    # Wrapper for the _pack_data method for this specific class
    def pack_data(self, file_name, reduce_mem_consumption: bool = True) -> dict:

        return self._pack_data(reduce_mem_consumption, file_name=file_name)

    
    def save(self, file_name):
        """
        Saves the class data to an HDF5 file.

        Args:
            file_name (str): The name of the file to save the data to.
        """
        h5f = self.pack_data(file_name=file_name)
        h5f.close()


    @classmethod
    def unpack(cls, file_name):
        """
        Loads class data from an HDF5 file and returns a class instance.

        Args:
            file_name (str): The name of the file to load the data from.

        Returns:
            discrete_brute_scan_hyperparameter_likelihood: An instance of the class with the loaded data.
        """
        with h5py.File(file_name, 'r') as h5f:
            # Create a new instance of the class
            class_input_dict = {}

            # Load simple attributes
            class_input_dict['log_likelihoodnormalisation'] = np.array(h5f['log_likelihoodnormalisation'])
            class_input_dict['log_marginalisation_regularisation'] = float(h5f.attrs['log_marginalisation_regularisation'])

            # Load numpy arrays
            if 'axes' in h5f:
                class_input_dict['axes'] = [np.array(h5f['axes'][str(i)][:]) for i in range(len(h5f['axes']))]

            if 'nuisance_axes' in h5f:
                class_input_dict['nuisance_axes'] = [np.array(h5f['nuisance_axes'][str(i)][:]) for i in range(len(h5f['nuisance_axes']))]

            if 'log_margresults' in h5f:
                class_input_dict['log_margresults'] = np.array(h5f['log_margresults'])

            if 'log_hyperparameter_likelihood' in h5f:
                class_input_dict['log_hyperparameter_likelihood'] = np.array(h5f['log_hyperparameter_likelihood'])

            if 'log_posterior' in h5f:
                class_input_dict['log_posterior'] = np.array(h5f['log_posterior'])

            # Load ParameterSet objects
            # Assuming you have a method in ParameterSet to load from a group
            if 'prior_param_set' in h5f:
                class_input_dict['parameter_specifications'] = []
                for prior_name in h5f['prior_param_set']:
                    prior_parameter_set = ParameterSet.load(h5f=h5f['prior_param_set'][prior_name])
                    class_input_dict['parameter_specifications'].append(prior_parameter_set)

            if 'mixture_param_specifications' in h5f:
                mixture_param_specifications_group = h5f['mixture_param_specifications']
                class_input_dict['mixture_param_specifications'] = ParameterSet.load(h5f=mixture_param_specifications_group)

        

            return class_input_dict
        
    @classmethod
    def load(cls, file_name, *args, **kwargs):
        class_input_dict = cls.unpack(file_name)
        instance = cls(*args, **class_input_dict, **kwargs)

        return instance

        



