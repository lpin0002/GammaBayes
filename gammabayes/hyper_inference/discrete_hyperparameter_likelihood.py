import numpy as np
from scipy.special import logsumexp
import functools
from tqdm import tqdm
from gammabayes.utils import angularseparation, convertlonlat_to_offset, iterate_logspace_simps, logspace_simpson
from gammabayes.utils.config_utils import save_config_file
from multiprocessing.pool import ThreadPool as Pool
import json, os, warnings
import matplotlib.pyplot as plt

class discrete_hyperparameter_likelihood(object):
    def __init__(self, priors, likelihood, axes=None,
                 dependent_axes=None, dependent_logjacob=0, 
                 hyperparameter_axes=(), numcores=8, 
                 likelihoodnormalisation= (), log_margresults=None, mixture_axes = None,
                 log_posterior=0, iterative_logspace_integrator=iterate_logspace_simps,
                 prior_matrix_list=None):
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

            hyperparameter_axes (tuple, optional): Tuple containing the default 
                values at which the priors will be evaluated. For example, if 
                there are two priors there will be two tuples each containing
                the range of hyperparameters at which each prior will be 
                evaluated at. Defaults to ().

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

            log_hyperparameter_likelihoods (np.ndarray, optional): A numpy array
                containing the log of hyperparameter marginalisation results. 
                Defaults to 0.

            log_posterior (np.ndarray, optional): A numpy array containing the
                log of hyperparameter posterior results. Defaults to 0.
        """
        
        self.priors                     = priors
        self.likelihood                 = likelihood
        self.axes                       = axes
            
        if dependent_axes is None:
            self.dependent_axes             = self.likelihood[0].dependent_axes
        else:
            self.dependent_axes             = dependent_axes
            
        self.dependent_logjacob         = dependent_logjacob
        self.hyperparameter_axes        = hyperparameter_axes
        self.numcores                   = numcores
        self.likelihoodnormalisation    = likelihoodnormalisation
        self.log_margresults               = log_margresults
        if mixture_axes is None:
            self.mixture_axes               = np.array([np.linspace(0,1,101)]*(len(priors)-1))
        elif len(mixture_axes) != len(priors):
            self.mixture_axes               = np.array([mixture_axes]*len(priors))
        else:
            self.mixture_axes = np.array([*mixture_axes])


        self.log_posterior = log_posterior
        self.iterative_logspace_integrator = iterative_logspace_integrator

        self.prior_matrix_list = prior_matrix_list
        
    
    def observation_nuisance_marg(self, axisvals, prior_matrix_list):
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
        
        
        likelihoodvalues = self.likelihood(*flattened_meshvalues).reshape(meshvalues[0].shape)
        
        likelihoodvalues = likelihoodvalues - self.likelihoodnormalisation
        
        all_log_marg_results = []
        for idx, prior_matrices in enumerate(prior_matrix_list):
            # single_parameter_log_margvals = []
            prior_matrices = np.asarray(prior_matrices)
            # for idx2, single_prior_matrix in enumerate(prior_matrices):
            logintegrandvalues = np.squeeze(prior_matrices+self.dependent_logjacob+likelihoodvalues)

            likelihoodvalues_ndim = np.squeeze(likelihoodvalues).ndim
            prior_matrices_ndim = np.squeeze(prior_matrices).ndim
            axisindices = np.arange(prior_matrices_ndim-likelihoodvalues_ndim, prior_matrices_ndim)

            output = self.iterative_logspace_integrator(logintegrandvalues,   
                axes=self.dependent_axes, axisindices=axisindices, )

            single_parameter_log_margvals = output

            all_log_marg_results.append(np.squeeze(np.asarray(single_parameter_log_margvals)))
            
            
        return np.array(all_log_marg_results, dtype=object)


    def nuisance_log_marginalisation(self, axisvals):

        
        # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
        np.seterr(divide='ignore', invalid='ignore')
        # log10eaxistrue,  longitudeaxistrue, latitudeaxistrue = axes
        
        nans = 0
        if self.prior_matrix_list is None:
            print("prior_matrix_list does not exist. Constructing priors.")
            prior_matrix_list = []
            for idx, prior in tqdm(enumerate(self.priors), total=len(self.priors), desc='Setting up prior matrices'):
                prior_matrices = []

                hyper_parameter_coords = np.asarray(np.meshgrid(*self.hyperparameter_axes[idx], indexing='ij'))

                flattened_hyper_parameter_coords = np.asarray([mesh.flatten() for mesh in hyper_parameter_coords]).T

                prior_matrices = np.empty(shape=(flattened_hyper_parameter_coords.shape[0],*np.squeeze(self.likelihoodnormalisation).shape,))
                for idx, hyperparametervalue in enumerate(flattened_hyper_parameter_coords):
                    prior_matrix = np.squeeze(prior.construct_prior_array(hyperparameters=hyperparametervalue, normalise=True))
                    nans+=np.sum(np.isnan(prior_matrix))
                    prior_matrices[idx,...] = prior_matrix


                prior_matrices = prior_matrices.reshape(tuple(list(hyper_parameter_coords[0].shape)+list(prior_matrices[0].shape)))

                prior_matrix_list.append(prior_matrices)

            print(f"Total cumulative number of nan values within all prior matrices: {nans}")

                
            self.prior_matrix_list = prior_matrix_list
        marg_partial = functools.partial(self.observation_nuisance_marg, prior_matrix_list=self.prior_matrix_list)

        with Pool(self.numcores) as pool:
            margresults = pool.map(marg_partial, zip(*axisvals))
        
        return margresults
    
    def apply_direchlet_stick_breaking_direct(self, xi_axes, depth):
        direchletmesh = 1

        for i in range(depth):
            direchletmesh*=(1-xi_axes[i])
        if depth!=len(xi_axes):
            direchletmesh*=xi_axes[depth]

        return direchletmesh
    
    
    
    def add_log_nuisance_marg_results(self, new_log_marg_results):
        """Add log nuisance marginalisation results to those within the class.

        Args:
            new_log_marg_results (np.ndarray): The log likelihood values after 
                marginalising over nuisance parameters.
        """
        self.log_margresults = np.append(self.log_margresults, new_log_marg_results, axis=0)
            
    def create_discrete_mixture_log_hyper_likelihood(self, mixture_axes=None, log_margresults=None):
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



        # reshape log_margresults into shape of (num_components, ...)
        reshaped_log_margresults = np.asarray(log_margresults)

        # Creating components of mixture for each prior
        mixture_array_list = []
        for idx in range(log_margresults.shape[1]):
            log_margresults_for_idx = np.vstack(reshaped_log_margresults[:,idx])
            mixcomp = np.expand_dims(np.log(
                self.apply_direchlet_stick_breaking_direct(xi_axes=mix_axes, depth=idx)), 
                axis=(*np.delete(np.arange(log_margresults_for_idx.ndim+len(mix_axes)), 
                                    np.arange(len(mix_axes))+1),)) 

            mixcomp=mixcomp+np.expand_dims(log_margresults_for_idx, 
                                        axis=(*(np.arange(len(mix_axes))+1),)
                                    )
            # print(idx, np.sum(np.where(np.isnan(mixcomp))))
            mixture_array_list.append(mixcomp)


        # Now to get around the fact that every component of the mixture can be a different shape
        # This should be the final output except for the dimension over which there are different observations (shown here as len(self.log_margresults))
        final_output_shape = [len(log_margresults), *np.arange(len(mixture_axes)), ]

        # Axes for each of the priors to __not__ expand in
        prioraxes = []
        # Counter for how many hyperparameter axes we have used
        hyper_idx = len(mixture_axes)+1
        for idx, hyperparameteraxes in enumerate(self.hyperparameter_axes):
            prior_axis_instance = list(range(1+len(mix_axes)))
            for hyperparameteraxis in hyperparameteraxes:
                try:
                    final_output_shape.append(hyperparameteraxis.shape[0])
                except:
                    final_output_shape.append(1)
                prior_axis_instance.append(hyper_idx)
            hyper_idx+=1
            prioraxes.append(prior_axis_instance)


        # Making all the mixture components the same shape that is compatible for adding together (in logspace)
        for prior_idx, mixture_array in enumerate(mixture_array_list):
            axis = []
            for i in range(len(final_output_shape)):
                if not(i in prioraxes[prior_idx]):
                    axis.append(i)
            axis = tuple(axis)
            mixture_array   =   np.expand_dims(mixture_array, axis=axis)
            mixture_array_list[prior_idx] = mixture_array


        # Now to combine the results for all the data and to 
        combined_mixture = -np.inf
        for idx, mixture_component in enumerate(mixture_array_list):
            combined_mixture = np.logaddexp(combined_mixture, mixture_component)

        unnormed_log_posterior = np.sum(combined_mixture, axis=0)

        # # Saving the result to the object
        # self.unnormed_log_posterior = unnormed_log_posterior

        return unnormed_log_posterior
        

    def combine_hyperparameter_likelihoods(self, log_hyperparameter_likelihoods):
        """To combine log hyperparameter likelihoods from multiple runs by 
            adding the resultant log_hyperparameter_likelihoods together.

        Args:
            log_hyperparameter_likelihoods (np.ndarray): Hyperparameter 
                log-likelihood results from a separate run with the same hyperparameter axes.
        """
        self.log_hyperparameter_likelihoods += log_hyperparameter_likelihoods

    def apply_uniform_hyperparameter_log_priors(self, priorinfo):
        """A function to apply uniform log priors for the given prior information.

        Format for each set of prior information within the tuple should be
        priorinfo[0] = minimum value of hyperparameter range
        priorinfo[1] = maximum value of hyperparameter range
        priorinfo[2] = hyperparameter range spacing
        priorinfo[3] = number of entries sampled within the hyperparameter range

        And each entry of the tuple then corresponds to each hyperparameter
        Args:
            priorinfo_tuple (tuple): A tuple containing the min, max and spacing
                of the hyperparameter axes sampled to generate the discrete 
                hyperparameter log likelihood
        Returns:
            array like: The natural log of the discrete hyperparameter uniform 
                prior values
        """
        priormesh = 0
        priorval_list = []
        for prior_info in priorinfo:
            hyper_prioraxis = np.arange(prior_info[0], prior_info[1], prior_info[2])
            hyper_priorvals = hyper_prioraxis*0+1/len(hyper_prioraxis)
            priorval_list.append(hyper_priorvals[:priorinfo[3]])

        hyper_priormesh = np.meshgrid(*priorval_list, indexing='ij')

        self.log_posterior = self.log_hyperparameter_likelihoods+np.log(hyper_priormesh)

        return np.log(hyper_priormesh)

            
            
    def save_data(self, directory_path='', reduce_mem_consumption=True):
        """A method to save the information contained within a class instance.

        This method saves all the attributes of a class instance to a dictionary 
            and then pickles it to the specified directory path. By default the 
            input priors and likelihoods are not saved to save on memory, but if 
            this is not an issue and you wish to save these then set 
            reduce_data_consumption to False.

        Args:
            directory_path (str, optional): Path to where to save class data. 
            Defaults to ''.

            reduce_mem_consumption (bool, optional): A bool parameter of whether
                to not save the input prior and likelihoods to save on memory 
                consumption. Defaults to True (meaning to __not__ save the prior 
                and likelihood).
        """
        data_to_save = {}
        
        data_to_save['hyperparameter_axes']                 = self.hyperparameter_axes
        if not(reduce_mem_consumption):
            data_to_save['priors']                              = self.priors
            data_to_save['likelihood']                          = self.likelihood
            
        data_to_save['dependent_logjacob']                  = self.dependent_logjacob
        data_to_save['axes']                                = self.axes
        data_to_save['dependent_axes']                      = self.dependent_axes
        data_to_save['marg_results']                        = self.marg_results
        data_to_save['mixture_axes']                        = self.mixture_axes
        data_to_save['log_hyperparameter_likelihoods']      = self.log_hyperparameter_likelihoods
        data_to_save['log_posterior']                       = self.log_posterior
        
                
        save_file_path = os.path.join(directory_path, "hyper_parameter_data.yaml")
        
        
        try:
            save_config_file(data_to_save, save_file_path)
        except:
            warnings.warn("Something went wrong when trying to save to the specified directory. Saving to current working directory")
            save_config_file("hyper_parameter_data.yaml", save_file_path)
        