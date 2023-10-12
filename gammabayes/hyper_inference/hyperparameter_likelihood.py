import numpy as np
from scipy.special import logsumexp
import functools
from tqdm.auto import tqdm
from ..utils.utils import angularseparation, convertlonlat_to_offset
from ..utils.plotting import mixture_posterior_plot
from multiprocessing.pool import ThreadPool as Pool
import os, warnings, pickle

class hyperparameter_likelihood(object):
    def __init__(self, priors=None, likelihood=None, axes=None,
                 dependent_axes=None, dependent_logjacob=0, 
                 hyperparameter_axes=(), numcores=8, 
                 likelihoodnormalisation= 0, log_marg_results=None, mixture_axes = None,
                 log_hyperparameter_likelihoods=0, log_posterior=0):
        """Initialise a hyperparameter_likelihood class instance.

        Args:
            priors (tuple, optional): Tuple containing instances of the 
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

            dependent_logjacob (np.ndarray or float, optional): A matrix of
                log jacobian values used during marginalisation, must be either 
                a single float value or if the depepdent axes are shapes 
                (m1,), (m2,),..., (mn,) then the dependent_logjacob must be of
                the shape (m1, m2,..., mn). Defaults to 0.

            hyperparameter_axes (tuple, optional): _description_. Defaults to ().

            numcores (int, optional): _description_. Defaults to 8.

            likelihoodnormalisation (np.ndarray or float, optional): _description_. Defaults to ().

            log_marg_results (np.ndarray, optional): _description_. Defaults to None.

            mixture_axes (tuple, optional): _description_. Defaults to None.

            log_hyperparameter_likelihoods (np.ndarray, optional): _description_. Defaults to 0.

            log_posterior (np.ndarray, optional): _description_. Defaults to 0.
        """
        
        self.priors                         = priors
        self.likelihood                     = likelihood
        self.axes                           = axes
            
        if dependent_axes is None:
            try:
                self.dependent_axes         = self.likelihood[0].dependent_axes
            except:
                pass
        else:
            self.dependent_axes             = dependent_axes
            
        self.dependent_logjacob             = dependent_logjacob
        self.hyperparameter_axes            = hyperparameter_axes
        self.numcores                       = numcores
        self.likelihoodnormalisation        = likelihoodnormalisation

        self.log_marg_results               = log_marg_results

        self.mixture_axes = mixture_axes

        self.log_posterior = log_posterior
        self.log_hyperparameter_likelihoods = log_hyperparameter_likelihoods

    def init_from_pickle(self, hyperparameter_pickle):
        """Sets all the possible attributes for a hyperparameter_likelihood 
            instance from saved hyperparameter pickle data.

        Args:
            hyperparameter_pickle (dict): A dict containing relevant 
                hyperparameter_likelihood parameters
        """

        try:
            self.priors                     = hyperparameter_pickle['priors']
        except:
            pass

        try:
            self.likelihood                 = hyperparameter_pickle['likelihood']
        except:
            pass

        try:
            self.axes                       = hyperparameter_pickle['axes']
        except:
            pass

        try:
            self.dependent_axes             = hyperparameter_pickle['dependent_axes']
        except:
            pass

        try:
            self.dependent_logjacob         = hyperparameter_pickle['dependent_logjacob']
        except:
            pass
        try:
            self.hyperparameter_axes        = hyperparameter_pickle['hyperparameter_axes']
        except:
            pass
        try:
            self.numcores                   = hyperparameter_pickle['numcores']
        except:
            pass
        try:
            self.likelihoodnormalisation    = hyperparameter_pickle['likelihoodnormalisation']
        except:
            pass
        try:
            self.log_marg_results           = hyperparameter_pickle['log_marg_results']
        except:
            pass
        try:
            self.mixture_axes               = hyperparameter_pickle['mixture_axes']
        except:
            pass
        try:
            self.log_hyperparameter_likelihoods = hyperparameter_pickle['log_hyperparameter_likelihoods']
        except:
            pass
        try:
            self.log_posterior              = hyperparameter_pickle['log_posterior']
        except:
            pass



        
    def construct_prior_arrays_over_hyper_axes(self, prior, 
                               dependent_axes, dependent_logjacob,
                               hyperparameter_axes):
        """Returns the log prior matrix for the given hyperparameter values.

        Args:
            prior (discrete_logprior): An instance of the discrete_logprior 
                GammaBayes class that contains the wanted prior.

            dependent_axes (tuple): A tuple over which the prior will be 
                evaluated.

            dependent_logjacob (np.ndarray): Natural log of the jacobian for 
                normalisation.

            hyperparameter_axes (tuple): Tuple of hyperparameter values to 
                construct the log prior matrix with.

        Returns:
            np.ndarray: matrix of the log prior values over the given axes.
        """
        if dependent_axes is None:
            dependent_axes      = self.dependent_axes

        if dependent_logjacob is None:
            dependent_logjacob  = self.dependent_logjacob

        if hyperparameter_axes is None:
            if not(self.hyperparameter_axes is None):
                return prior.construct_prior_array(self.hyperparameter_axes)
            
            else:
                return (prior.construct_prior_array(), )
        else:
            return prior.construct_prior_array(hyperparameter_axes)


    
    def observation_nuisance_marg(self, axisvals, signal_prior_matrices, logbkgpriorvalues):
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

            signal_prior_matrices (list): A list of the log prior matrices for
                the various values of the relevant hyperparameters for the 
                signal prior.

            logbkgpriorvalues (np.ndarray): A matrix containing the log prior
                values for the background prior.

        Returns:
            np.ndarray: A numpy array of two elements, the first being a list
                of all the log marginalisation values for the various log prior
                matrices and the second element is the log marginalisation
                value for the log background prior matrix.
        """
        sig_log_marg_results = []
        
        meshvalues          = np.meshgrid(*axisvals, *self.dependent_axes, indexing='ij')
        flattened_meshvalues = [meshmatrix.flatten() for meshmatrix in meshvalues]
        
        
        likelihoodvalues    = self.likelihood(*flattened_meshvalues).reshape(meshvalues[0].shape)
        
        likelihoodvalues    = likelihoodvalues - self.likelihoodnormalisation
        
        bkg_logmargresult   = logsumexp(logbkgpriorvalues+self.dependent_logjacob+likelihoodvalues)

        
        singlemass_sigmargvals = []
        for logmass_idx, signal_prior_matrix in enumerate(signal_prior_matrices):
            
            output          = logsumexp(signal_prior_matrix+self.dependent_logjacob+likelihoodvalues)
            singlemass_sigmargvals.append(output)

        sig_log_marg_results.append(singlemass_sigmargvals)
            
            
        return np.array([np.squeeze(np.array(sig_log_marg_results)), bkg_logmargresult], dtype=object)


    def nuisance_log_marginalisation(self, axisvals):
        """Returns log of the nuisance marginalisation values

        A function that iterates through axisvals arguments, presumed to be sets
            of observations of gamma-ray events, and evaluates the log of the 
            resultant marginalisations over the gamma-ray event true values for 
            the class instances priors and observation likelihoods.

        Args:
            axisvals (subscriptable object): A tuple or list that contains the 
                axis values to be used during the nuisance marginalisation

        Returns:
            np.ndarray: A list containing the log of the nuisance marginalisation 
                values for the priors and likelihoods within the class.
        """

        
        # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
        np.seterr(divide='ignore', invalid='ignore')

        log_prior_matrix_list   = []
        for idx, prior in tqdm(enumerate(self.priors), total=len(self.priors), desc='Setting up prior matrices', ncols=80):
            log_prior_matrices  = []

            for hyperparametervalue in zip(*np.meshgrid(*self.hyperparameter_axes[idx])):
                log_priorvals   = np.squeeze(prior.construct_prior_array(hyperparameters=(hyperparametervalue,), normalise=True))
                log_prior_matrices.append(log_priorvals)

            log_prior_matrix_list.append(log_prior_matrices)
            

        log_marg_partial = functools.partial(self.observation_nuisance_marg, signal_prior_matrices=log_prior_matrix_list[0], logbkgpriorvalues=log_prior_matrix_list[1])

        with Pool(self.numcores) as pool:
                    log_marg_results = pool.map(log_marg_partial, 
                                                tqdm(zip(*axisvals), 
                                                     total=len(list(axisvals[0])), 
                                                     desc='Performing parallelized direct event marginalisation'))

        self.log_marg_results           = log_marg_results
        
        
        return log_marg_results
    
    
    
    def add_log_nuisance_marg_results(self, new_log_marg_results):
        """Add log nuisance marginalisation results to those within the class.

        Args:
            new_log_marg_results (np.ndarray): The log likelihood values after 
                marginalising over nuisance parameters.
        """
        if self.log_marg_results is None:
            self.log_marg_results       = new_log_marg_results
        else:
            self.log_marg_results       = np.append(self.log_marg_results, 
                                                    new_log_marg_results, axis=0)

        
            
    def create_mixture_log_hyperparameter_likelihood(self, mixture_axes=None, log_marg_results=None, hyperparameter_axes=None):
        
        
        if mixture_axes is None:
            mixture_axes                = self.mixture_axes
            if mixture_axes is None:
                raise Exception("Mixture axes not specified")
        else:
            self.mixture_axes           = mixture_axes

        if not(self.priors is None):
            if len(mixture_axes)!=len(self.priors):
                raise Exception(f""""Number of mixture axes does not match number of components. Please check your inputs.
                                Number of mixture axes is {len(mixture_axes)}
                                and number of prior components is {len(self.priors)}.""")
            
        if log_marg_results is None:
            log_marg_results            = self.log_marg_results

        if hyperparameter_axes is None:
            hyperparameter_axes         = self.hyperparameter_axes


        # Relative weights stay the same but now they will be 
        mixture_axes = mixture_axes/np.sum(mixture_axes, axis=0)

        # reshape log_marg_results into shape of (num_components, ...)
        reshaped_log_marg_results = np.asarray(log_marg_results).T
        print(f"Reshaped mixture shape: {reshaped_log_marg_results.shape}")

        # Creating components of mixture for each prior
        mixture_array_list = []
        for idx, mixture_array in enumerate(mixture_axes):
            log_marg_results_for_idx = np.vstack(reshaped_log_marg_results[idx,:])
            mixture_array_list.append(np.expand_dims(np.log(mixture_array), axis=(*(np.arange(log_marg_results_for_idx.ndim)+1),))+log_marg_results_for_idx[np.newaxis, ...])


        # Now to get around the fact that every component of the mixture can be a different shape
        # This should be the final output except for the dimension over which there are different observations (shown here as len(self.log_marg_results))
        final_output_shape = [*[mixture_array.shape for mixture_array in mixture_axes], len(log_marg_results)]
        final_output_shape.pop(0)

        # Axes for each of the priors to __not__ expand in
        prioraxes = []
        # Counter for how many hyperparameter axes we have used
        hyper_idx = 0
        for idx, hyperparameteraxes in enumerate(self.hyperparameter_axes):
            prior_axis_instance = list(np.arange(len(mixture_axes)))
            for hyperparameteraxis in hyperparameteraxes:
                try:
                    final_output_shape.append(hyperparameteraxis.shape[0])
                    
                except:
                    final_output_shape.append(1)
                prior_axis_instance.append(hyper_idx+2)
                hyper_idx+=1
            prioraxes.append(prior_axis_instance)


        # Making all the mixture components the same shape that is compatible for adding together (in logspace)
        for prior_idx, mixture_array in enumerate(mixture_array_list):
            axis = []
            for i in range(len(final_output_shape)):
                if not(i in prioraxes[prior_idx]):
                    axis.append(i)
                    
            axis = tuple(axis)

            mixture_array                   = np.expand_dims(mixture_array, axis=axis)

            mixture_array_list[prior_idx]   = mixture_array


        # Now to combine the results for all the data and to 
        combined_mixture = -np.inf
        for mixture_component in mixture_array_list:
            combined_mixture                = np.logaddexp(combined_mixture, mixture_component)

        log_hyperparameter_likelihoods      = np.sum(combined_mixture, axis=1)


        return log_hyperparameter_likelihoods

    def combine_hyperparameter_likelihoods(self, log_hyperparameter_likelihoods):
        """To combine log hyperparameter likelihoods from multiple runs by 
            adding the resultant log_hyperparameter_likelihoods together.

        Args:
            log_hyperparameter_likelihoods (np.ndarray): Hyperparameter 
                log-likelihood results from a separate run with the same hyperparameter axes.
        """
        self.log_hyperparameter_likelihoods += log_hyperparameter_likelihoods
            
            
    def save_data(self, directory_path='', reduce_data_consumption=True):
        """A method to save the information contained within a class instance.

        This method saves all the attributes of a class instance to a dictionary 
            and then pickles it to the specified directory path. By default the 
            input priors and likelihoods are not saved to save on memory, but if 
            this is not an issue and you wish to save these then set 
            reduce_data_consumption to False.

        Args:
            directory_path (str, optional): Path to which. Defaults to ''.

            reduce_data_consumption (bool, optional): _description_. 
                Defaults to True.
        """
        
        data_to_save = {}
        
        data_to_save['hyperparameter_axes']             = self.hyperparameter_axes
        if not(reduce_data_consumption):
            data_to_save['priors']                      = self.priors
            data_to_save['likelihood']                  = self.likelihood
            
        data_to_save['dependent_logjacob']              = self.dependent_logjacob
        data_to_save['axes']                            = self.axes
        data_to_save['dependent_axes']                  = self.dependent_axes
        data_to_save['log_marg_results']                = self.log_marg_results
        data_to_save['mixture_axes']                    = self.mixture_axes
        data_to_save['log_hyperparameter_likelihoods']  = self.log_hyperparameter_likelihoods

        data_to_save['log_posterior']                   = self.log_posterior
                
        save_file_path = directory_path+"/hyper_parameter_data.pkl"
        print(save_file_path)
        
        with open(save_file_path, "wb") as file:
            pickle.dump(data_to_save, file)

    def plot_posterior(self, log_posterior=None, hyperparameter_axes=None, mixture_axes=None, config_file=None):
        """A wrapper for the mixture_posterior_plot function.

        Args:
            log_posterior (np.ndarray, optional): The log of the posterior over 
                the hyperparameter axes (and mixture axes, although they are also 
                hyperparameters). Defaults to None.

            hyperparameter_axes (np.ndarray, optional): Logmass axis the 
                posterior is evaluated at. Defaults to None.

            mixture_axes (np.ndarray, optional): Mixture axis which the 
                posterior is defined over. Defaults to None.

            config_file (dict, optional): A dictionary containing run parameters
                such as the true values of the parameters and number of events. 
                Defaults to None.
        """
        if log_posterior is None:
            if type(self.log_posterior) != int:
                log_posterior       = self.log_posterior
            else:
                try:
                    log_posterior = self.log_hyperparameter_likelihoods
                except:
                    raise Exception("What are we plotting here?")

        if hyperparameter_axes is None:
            hyperparameter_axes = self.hyperparameter_axes[0][0]

        if mixture_axes is None:
            mixture_axes        = self.mixture_axes[0]

        print(f"Shapes: {log_posterior.shape}")
        print(f"{hyperparameter_axes.shape}")
        print(f"{mixture_axes.shape}")

        mixture_posterior_plot(log_posterior=np.squeeze(log_posterior), xi_range=mixture_axes, logmassrange=hyperparameter_axes, config_file=config_file)

        