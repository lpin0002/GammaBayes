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
from gammabayes.hyper_inference.utils import _handle_parameter_specification, _handle_nuisance_axes
from gammabayes.hyper_inference.mixture_scan_nuisance_scan_output import ScanOutput_ScanMixtureFracPosterior
from gammabayes.hyper_inference.mixture_sampling_nuisance_scan_output import ScanOutput_StochasticMixtureFracPosterior
from gammabayes import EventData, Parameter, ParameterSet
from gammabayes.priors import DiscreteLogPrior
from multiprocessing.pool import ThreadPool as Pool
import os, warnings, logging, time, h5py

class DiscreteBruteScan(object):
    def __init__(self, log_priors: list[DiscreteLogPrior] | tuple[DiscreteLogPrior] = None, 
                 log_likelihood: callable = None, 
                 axes: list[np.ndarray] | tuple[np.ndarray] | None=None,
                 nuisance_axes: list[np.ndarray] | tuple[np.ndarray] | None = None,
                 prior_parameter_specifications: dict | list[ParameterSet] | dict[ParameterSet] = {}, 
                 log_likelihoodnormalisation: np.ndarray | float = 0., 
                 log_nuisance_marg_results: np.ndarray | None = None, 
                 mixture_parameter_specifications: list[np.ndarray] | tuple[np.ndarray] | None = None,
                 log_hyperparameter_likelihood: np.ndarray | float = 0., 
                 log_posterior: np.ndarray | float = 0., 
                 iterative_logspace_integrator: callable = iterate_logspace_integration,
                 logspace_integrator: callable = logspace_riemann,
                 log_prior_matrix_list: list[np.ndarray] | tuple[np.ndarray] = None,
                 log_marginalisation_regularisation: float  = None,
                 no_priors_on_init: bool = False,
                 no_likelihood_on_init: bool = False,
                 mixture_fraction_exploration_type=None,
                 applied_priors=False,
                 ):
        """
        Initializes a discrete brute scan hyperparameter likelihood object.

        Args:
            log_priors (list[DiscreteLogPrior] | tuple[DiscreteLogPrior], optional): 
                Priors for the log probabilities of discrete input values. Defaults to None.
            
            log_likelihood (callable, optional): A callable object to compute 
                the log likelihood. Defaults to None.
            
            axes (list[np.ndarray] | tuple[np.ndarray] | None, optional): Axes 
                that the measured event data can take. Defaults to None.
            
            nuisance_axes (list[np.ndarray] | tuple[np.ndarray] | None, optional): 
                Axes that the nuisance parameters can take. Defaults to None.
            
            prior_parameter_specifications (dict, optional): Specifications for 
                parameters involved in the likelihood estimation. Defaults to an empty dictionary.
            
            log_likelihoodnormalisation (np.ndarray | float, optional): 
                Normalization for the log likelihood. Defaults to 0.
            
            log_nuisance_marg_results (np.ndarray | None, optional): Results of 
                marginalization, expressed in log form. Defaults to None.
            
            mixture_parameter_specifications (list[np.ndarray] | tuple[np.ndarray] | None, optional): 
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
        if not self.no_priors_on_init:
            self.nuisance_axes     = _handle_nuisance_axes(nuisance_axes,
                                                        log_likelihood=self.log_likelihood,
                                                        log_prior=self.log_priors[0])
            self.prior_parameter_specifications = _handle_parameter_specification(
                parameter_specifications=prior_parameter_specifications,
                num_required_sets=len(self.log_priors),
                _no_required_num=self.no_priors_on_init)
        else:
            self.nuisance_axes     = _handle_nuisance_axes(nuisance_axes,
                                                        log_likelihood=self.log_likelihood)
            self.prior_parameter_specifications = _handle_parameter_specification(
                parameter_specifications=prior_parameter_specifications,
                _no_required_num=self.no_priors_on_init)




        # Currently required as the normalisation of the IRFs isn't natively consistent
        self.log_likelihoodnormalisation            = np.asarray(log_likelihoodnormalisation)

        # Log-space integrator for multiple dimensions (kind of inefficient at the moment)
        self.iterative_logspace_integrator      = iterative_logspace_integrator

        # Single dimension log-space integrator (fully vectorised)
        self.logspace_integrator            = logspace_integrator

        # Doesn't have to be initialised here, but you can do it if you want
        self.mixture_parameter_specifications   = ParameterSet(mixture_parameter_specifications)

        # Used as regularisation to avoid
        self.log_marginalisation_regularisation = log_marginalisation_regularisation

        # From here it's the initialisation of results, but the arguments should be able
            # to be used as initialisation if you have produced you results in another
            # way, assuming that they match the expected inputs/outputs of what is
            # used/produce here.

        self.log_nuisance_marg_results                    = log_nuisance_marg_results
        self.log_hyperparameter_likelihood      = log_hyperparameter_likelihood
        self.log_posterior                      = log_posterior
        self.log_prior_matrix_list              = log_prior_matrix_list
        self.mixture_fraction_exploration_type  = mixture_fraction_exploration_type
        self.applied_priors = applied_priors



    # Most import method in this whole class. If debugging be firmly aware of
        # - prior matrices normalisations
        # - log_likelihood normalisations. 
            # Must be normalised over __measured/reconstructed__ axes __not__
            # over the dependent axes
        # - numerical stability
            # Tried my best to avoid this, but if input values imply real values 
            # below numerical precision, watch out
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

        for prior_idx, log_prior_matrices in enumerate(log_prior_matrix_list):
            logging.info(f"log prior number {prior_idx} shape: ", log_prior_matrices.shape)
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
                

                prior_parameter_specifications = self.prior_parameter_specifications[_prior_idx].scan_format

                prior_spectral_params   = prior_parameter_specifications['spectral_parameters']
                prior_spatial_params    = prior_parameter_specifications['spatial_parameters']


                prior_marged_shapes[_prior_idx] = (Nevents, 
                                                   *[parameter_specification.size for parameter_specification in prior_spectral_params.values()],
                                                   *[parameter_specification.size for parameter_specification in prior_spatial_params.values()])

                logging.info(f'prior_marged_shapes[{_prior_idx}]: ', prior_marged_shapes[_prior_idx])
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
                                             log_prior: DiscreteLogPrior, 
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
        diffs = log_marg_maxs-log_marg_mins
        self.log_marginalisation_regularisation = np.abs(0.3*np.mean(np.mean(diffs, axis=tuple(range(1, diffs.ndim)))))


        return reshaped_marg_results
    

    def add_log_nuisance_marg_results(self, new_log_marg_results: np.ndarray):
        """
        Extends each array in the existing log marginalisation results with corresponding new log marginalisation results. 
        This method iterates over each array in `self.log_nuisance_marg_results` and the new results, extending each existing array 
        with the corresponding new results. 

        Args:
            new_log_marg_results (np.ndarray): An array containing new log marginalisation results. Each element in this 
                                            array corresponds to and will be appended to the respective array in 
                                            `self.log_nuisance_marg_results`.

        Notes:
            - It's assumed that `self.log_nuisance_marg_results` is a list of lists (or arrays) where each sublist corresponds 
            to a set of log marginalisation results.
            - `new_log_marg_results` should have the same length/number of priors as `self.log_nuisance_marg_results` to ensure 
            correct pairing and extension of each sublist.
            - This method uses list comprehension to iterate and extend each corresponding sublist.
        """
        self.log_nuisance_marg_results = [log_margresult.extend(new_log_marg_result) for log_margresult, new_log_marg_result in zip(self.log_nuisance_marg_results, new_log_marg_results)]


    def select_scan_output_exploration_class(self, 
                                                       mixture_parameter_specifications: ParameterSet | list[Parameter] | dict,
                                                       mixture_fraction_exploration_type: str = None, 
                                                       log_nuisance_marg_results: list | np.ndarray = None,
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
            
            log_nuisance_marg_results (list, array like, optional): The logarithm of marginal results to be used in the exploration. If not provided, 
                defaults to the class attribute `log_nuisance_marg_results`.
            
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

        if log_nuisance_marg_results is None:
            log_nuisance_marg_results = self.log_nuisance_marg_results

        if mixture_fraction_exploration_type.lower() == 'scan':
            hyperspace_analysis_class = ScanOutput_ScanMixtureFracPosterior

        elif mixture_fraction_exploration_type.lower() == 'sample':
            hyperspace_analysis_class = ScanOutput_StochasticMixtureFracPosterior
            self.applied_priors = True

        else:
           raise ValueError("Invalid 'mixture_fraction_exploration_type' must be either 'scan' or 'sample'.")
        
        if self.mixture_parameter_specifications.dict_of_parameters_by_name == {}:
            logging.info("Setting 'self.mixture_parameter_specifications' to that specified in 'select_scan_output_exploration_class'. ")
            self.mixture_parameter_specifications = mixture_parameter_specifications
        
        self.hyper_analysis_instance = hyperspace_analysis_class(
                log_nuisance_marg_results = log_nuisance_marg_results,
                log_nuisance_marg_regularisation = self.log_marginalisation_regularisation,
                mixture_parameter_specifications=mixture_parameter_specifications,
                prior_parameter_specifications=prior_parameter_specifications,
                *args, **kwargs)

        

    def init_posterior_exploration(self, *args, **kwargs):
        """
        Initiates the posterior exploration process.
        
        This method delegates the initiation of exploration to the instance of the exploration class selected by the 
        `select_scan_output_exploration_class` method. It prepares the exploration environment and parameters 
        based on the class instance's configuration.
        
        *args, **kwargs: Arguments and keyword arguments to be passed to the initiation method of the exploration class.
        """
        self.hyper_analysis_instance.initiate_exploration(*args, **kwargs)

    def run_posterior_exploration(self, *args, **kwargs):
        """
        Runs the posterior exploration process and returns the results.
        
        This method triggers the actual exploration process using the selected and initialized exploration class instance. 
        It runs the exploration based on the configured parameters and returns the exploration results.
        
        *args, **kwargs: Arguments and keyword arguments to be passed to the run method of the exploration class.
        
        Returns:
            The results of the posterior exploration, the format and content of which depend on the exploration method used.
        """
        self._posterior_exploration_output = self.hyper_analysis_instance.run_exploration(*args, **kwargs)

        if self.applied_priors and (self.mixture_fraction_exploration_type.lower() == 'scan'):
            self.log_posterior = self._posterior_exploration_output

        elif not self.applied_priors and (self.mixture_fraction_exploration_type.lower() == 'scan'):

            self.log_hyperparameter_likelihood = self._posterior_exploration_output

            self.apply_hyperparameter_priors()

        else:
            self.log_posterior = self.hyper_analysis_instance.sampler.results

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
            result = self._posterior_exploration_output
            if self.applied_priors:
                self.log_posterior = result
            else:
                self.log_hyperparameter_likelihood = result

            return result
        else:
            results = self.hyper_analysis_instance.sampler.results

            self.log_posterior = results.samples

            return results
        

    
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
    

    def apply_hyperparameter_priors(self, 
                                    log_hyperparameter_likelihood_matrix=None,
                                    prior_parameter_specifications: list[ParameterSet] | dict[dict[dict[dict]]] | list[dict[dict[dict]]] = None, 
                                    mixture_parameter_specifications: list[Parameter] | ParameterSet = None,
                                    log_hyper_priormesh: np.ndarray = None, 
                                    integrator: callable = None
                                    ):
        """
        Applies uniform priors to hyperparameters and calculates the posterior using the updated likelihood and prior mesh.

        Assumes hyperparameter likelihood is a scan output/matrix ____not samples____, sample outputs are presumed to have priors
        inerherently applied with the use of the relevant sampler.

        This method is designed to compute the posterior distribution of hyperparameters by applying uniform priors, 
        generating a meshgrid of log prior values, and combining it with the log hyperparameter likelihood.

        Args:
            priorinfos (list[dict] | tuple[dict]): Information about the priors, such as range and resolution.
            hyper_param_axes (list[np.ndarray] | tuple[np.ndarray], optional): Axes for the hyperparameters, used to 
                generate the meshgrid for priors. If None, it must be computed from `priorinfos`.
            log_hyper_priormesh (np.ndarray, optional): Pre-computed log prior meshgrid. If None, it is computed from 
                `priorinfos` and `hyper_param_axes`.
            integrator (callable, optional): Integrator function to be used for calculating the posterior. Defaults to 
                `self.logspace_integrator` if None.

        Returns:
            tuple: Contains the log posterior, list of log prior values, and the hyperparameter values lists. Specifically,
            (log_posterior, log_prior_val_list, hyper_val_list).
        """
        
        if integrator is None:
            integrator = self.logspace_integrator

        if log_hyperparameter_likelihood_matrix is None:
            log_hyperparameter_likelihood_matrix = self.log_hyperparameter_likelihood
        
        if prior_parameter_specifications is None:
            prior_parameter_specifications = self.prior_parameter_specifications

        else:
            prior_parameter_specifications = _handle_parameter_specification(
                parameter_specifications=prior_parameter_specifications,
                num_required_sets=len(self.log_priors),)
            

        if mixture_parameter_specifications is None:
            mixture_parameter_specifications = self.mixture_parameter_specifications

        else:
            mixture_parameter_specifications = ParameterSet(mixture_parameter_specifications)

            
        if log_hyper_priormesh is None:
            prior_parameter_axes = []
            log_prior_param_probabilities = []


            for mixture_parameter_name, mixture_parameter in mixture_parameter_specifications.items():
                mixture_parameter_axis = mixture_parameter.axis
                mixture_parameter_log_probability_axis = mixture_parameter.logpdf(mixture_parameter_axis)
                prior_parameter_axes.append(mixture_parameter_axis)
                log_prior_param_probabilities.append(mixture_parameter_log_probability_axis)


            for prior_parameters in prior_parameter_specifications:
                for parameter_name, parameter in prior_parameters.items():
                    parameter_axis = parameter.axis
                    parameter_log_probability_axis = parameter.logpdf(parameter_axis)
                    prior_parameter_axes.append(parameter_axis)
                    log_prior_param_probabilities.append(parameter_log_probability_axis)



            log_hyper_priormeshes = np.meshgrid(*log_prior_param_probabilities, indexing='ij')
            log_hyper_priormesh = np.sum(log_hyper_priormeshes, axis=0)


        self.log_posterior = np.squeeze(log_hyperparameter_likelihood_matrix)+log_hyper_priormesh

        self.applied_priors = True

        return self.log_posterior, log_prior_param_probabilities, prior_parameter_axes

    # Private method for classes that inherit this classes behaviour åto use
        # and don't have to re-write many of the same lines again
    def _pack_data(self, reduce_mem_consumption: bool = True, h5f=None, file_name='temporary_file.h5') -> dict:
        """
        Packs class data into an HDF5 file format for efficient storage and retrieval.

        This private method is designed for internal use by classes inheriting this behavior to avoid rewriting common 
        functionality. It efficiently stores attributes, numpy arrays, and ParameterSet objects into an HDF5 file.

        Args:
            reduce_mem_consumption (bool): If True, reduces memory consumption by not storing certain large data structures.
            
            h5f (h5py.File, optional): An open h5py.File object. If None, a new file is created using `file_name`.
            
            file_name (str): Name of the HDF5 file to create or write to if `h5f` is None.

        Returns:
            dict: The h5py.File object, which acts as a dictionary, containing all the stored data.
        
        Notes:
            - Users should ensure that `h5f` is properly closed after use to prevent data corruption.
        """
        if h5f is None:
            h5f = h5py.File(file_name, 'w-')  # Replace 'temporary_file.h5' with desired file name
        
        
        h5f.attrs['log_marginalisation_regularisation'] = self.log_marginalisation_regularisation
        h5f.attrs['mixture_fraction_exploration_type']  = self.mixture_fraction_exploration_type
        
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


        if self.log_nuisance_marg_results is not None:
            h5f.create_dataset('log_nuisance_marg_results', data=self.log_nuisance_marg_results)

        if self.log_hyperparameter_likelihood is not None:
            h5f.create_dataset('log_hyperparameter_likelihood', data=self.log_hyperparameter_likelihood)


        if self.log_posterior is not None:
            h5f.create_dataset('log_posterior', data=np.asarray(self.log_posterior, dtype=float))


        # Packing ParameterSet objects
        if self.prior_parameter_specifications is not None:
            prior_param_set_group = h5f.create_group('prior_param_set')

            for prior_idx, (prior, single_prior_param_set) in enumerate(zip(self.log_priors, self.prior_parameter_specifications)):
                if type(single_prior_param_set) == ParameterSet:
                    single_prior_param_group = prior_param_set_group.create_group(prior.name)
                    single_prior_param_group = single_prior_param_set.pack(h5f=single_prior_param_group)

        if self.mixture_parameter_specifications is not None:

            mixture_parameter_specifications_group = h5f.create_group('mixture_parameter_specifications')

            mixture_parameter_specifications_group = self.mixture_parameter_specifications.pack(h5f=mixture_parameter_specifications_group)


        # if not reduce_mem_consumption:

        #     if self.log_prior_matrix_list is not None:
        #         log_prior_matrix_list_group = h5f.create_group('log_prior_matrix_list')

        #         for prior, single_prior_log_matrix_list in zip(self.log_priors, self.log_prior_matrix_list):
        #             log_prior_matrix_list_group.create_dataset(prior.name, single_prior_log_matrix_list)


        return h5f
    

    # Wrapper for the _pack_data method for this specific class
    def pack_data(self, file_name:str, reduce_mem_consumption: bool = True) -> dict:
        """
        Public wrapper for the `_pack_data` method, facilitating data packing into an HDF5 file.

        Args:
            file_name (str): The name of the file where data should be packed.
            reduce_mem_consumption (bool, optional): If True, optimizes the packing process to consume less memory.

        Returns:
            dict: A reference to the HDF5 file object containing the packed data.
        """

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
        Class method that loads class data from an HDF5 file, creating a class instance with the loaded data.

        Args:
            file_name (str): The name of the HDF5 file from which to load the data.

        Returns:
            An instance of the class populated with the data loaded from the specified HDF5 file.
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

            if 'log_nuisance_marg_results' in h5f:
                class_input_dict['log_nuisance_marg_results'] = np.array(h5f['log_nuisance_marg_results'])

            if 'log_hyperparameter_likelihood' in h5f:
                class_input_dict['log_hyperparameter_likelihood'] = np.array(h5f['log_hyperparameter_likelihood'])

            if 'log_posterior' in h5f:
                class_input_dict['log_posterior'] = np.array(h5f['log_posterior'])

            # Load ParameterSet objects
            # Assuming you have a method in ParameterSet to load from a group
            if 'prior_param_set' in h5f:
                class_input_dict['prior_parameter_specifications'] = []
                for prior_name in h5f['prior_param_set']:
                    prior_parameter_set = ParameterSet.load(h5f=h5f['prior_param_set'][prior_name])
                    class_input_dict['prior_parameter_specifications'].append(prior_parameter_set)

            if 'mixture_parameter_specifications' in h5f:
                mixture_parameter_specifications_group = h5f['mixture_parameter_specifications']
                class_input_dict['mixture_parameter_specifications'] = ParameterSet.load(h5f=mixture_parameter_specifications_group)

        

            return class_input_dict
        
    @classmethod
    def load(cls, file_name:str, *args, **kwargs):
        """
        Class method that creates an instance of the class with data loaded from an HDF5 file.

        This method uses `unpack` to load data from the file and then initializes a class instance with the loaded data.

        Args:
            file_name (str): The name of the HDF5 file from which to load the data.
            *args: Additional positional arguments passed to the class constructor.
            **kwargs: Additional keyword arguments passed to the class constructor.

        Returns:
            An instance of the class with data initialized from the specified HDF5 file.
        """
        class_input_dict = cls.unpack(file_name)
        instance = cls(*args, **class_input_dict, **kwargs)

        return instance

        



