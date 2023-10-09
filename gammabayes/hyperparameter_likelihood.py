from scipy.special import logsumexp
import functools
from tqdm.auto import tqdm
from .utils.utils import angularseparation, convertlonlat_to_offset
from .plotting import plot_posterior
from multiprocessing.pool import ThreadPool as Pool
import json, os, warnings
import pickle, time

try:
    import cupy as xp
except:
    import numpy as xp

class hyperparameter_likelihood(object):
    def __init__(self, priors=None, likelihood=None, axes=None,
                 dependent_axes=None, dependent_logjacob=0, 
                 hyperparameter_axes_tuple=(), numcores=8, 
                 likelihoodnormalisation= (), log_margresults=None, mixture_axes = None, unnormed_log_posterior=0,
                 name='', priors_applied=None):
        
        self.priors                     = priors
        self.likelihood                 = likelihood
        self.axes                       = axes
        self.name                       = name

            
        if dependent_axes is None:
            try:
                self.dependent_axes         = self.likelihood[0].dependent_axes
            except:
                warnings.warn("No depedent axes given. Being assigned 'None'.")
        else:
            self.dependent_axes         = dependent_axes
            
        self.dependent_logjacob         = dependent_logjacob
        self.hyperparameter_axes_tuple        = hyperparameter_axes_tuple
        self.numcores                   = numcores
        self.likelihoodnormalisation    = likelihoodnormalisation
        self.log_margresults            = log_margresults

        if mixture_axes is None:
            if priors is None:
                warnings.warn("You're not giving me a lot to with?")
            else:
                self.mixture_axes           = xp.array([xp.linspace(0,1,101)]*(len(priors)-1))
        elif priors is None:
                warnings.warn("You're not giving me a lot to with?")
        elif len(mixture_axes) != len(priors):
            self.mixture_axes           = xp.array([mixture_axes]*len(priors))
        else:
            self.mixture_axes = xp.array([*mixture_axes])

        self.unnormed_log_posterior = unnormed_log_posterior
        if priors_applied is None:
            self.priors_applied = 0
        else:
            self.priors_applied = priors_applied
    
    def initiate_from_dict(self, dict):        
        
        try:
            self.priors         = dict['priors'] 
        except:
            warnings.warn("Input dictionary does not contain priors")

        try:
            self.likelihood     = dict['likelihood'] 
        except:
            warnings.warn("Input dictionary does not contain likelihoods")
        try:
            self.log_margresults           = dict['log_margresults']
        except:
            warnings.warn("Input dictionary does not contain log marginalisation results.")

        try: 
            self.priors_applied = dict['priors_applied']
        except:
            warnings.warn("""
Input dictionary does not contain whether prior values have been applied to the posterior.
    Assuming that it has not been applied.""")
            self.priors_applied = 0

        self.hyperparameter_axes_tuple    = dict['hyperparameter_axes_tuple'] 
        self.dependent_logjacob     = dict['dependent_logjacob']
        self.axes                   = dict['axes']
        self.dependent_axes         = dict['dependent_axes']
        
        self.unnormed_log_posterior              = dict['unnormed_log_posterior']


            
        self.mixture_axes               = dict['mixture_axes']
        self.unnormed_log_posterior     = dict['unnormed_log_posterior']


        
    def construct_prior_arrays_over_hyper_axes(self, prior, 
                               dependent_axes, dependent_logjacob,
                               hyperparameter_axes):
        if dependent_axes is None:
            dependent_axes = self.dependent_axes
        if dependent_logjacob is None:
            dependent_logjacob = self.dependent_logjacob
        if hyperparameter_axes is None:
            if not(self.hyperparameter_axes is None):
                return prior.construct_prior_array(self.hyperparameter_axes)
            else:
                return (prior.construct_prior_array(), )


    
    def observation_marg(self, axisvals, prior_matrix_list):


        
        meshvalues  = xp.meshgrid(*axisvals, *self.dependent_axes, indexing='ij')


        flattened_meshvalues = [meshmatrix.flatten() for meshmatrix in meshvalues]
        
        
        likelihoodvalues = self.likelihood(*flattened_meshvalues).reshape(meshvalues[0].shape)
        
        likelihoodvalues = likelihoodvalues - self.likelihoodnormalisation
        
        margresultlist = []
        for idx in range(len(prior_matrix_list)):
            single_prior_matrices = prior_matrix_list[idx]
            single_prior_matrix_results = []

            for single_prior_matrix in single_prior_matrices:

                output = logsumexp(single_prior_matrix+self.dependent_logjacob+likelihoodvalues)

                single_prior_matrix_results.append(output)
                
            margresultlist.append(single_prior_matrix_results)


        return margresultlist


    def full_obs_marginalisation(self, axisvals, parallelize=True):

        
        # Makes it so that when xp.log(0) is called a warning isn't raised as well as other warnings stemming from this.
        xp.seterr(divide='ignore', invalid='ignore')

        if self.priors is None:
            raise Exception("No priors given.")
        
        if self.likelihood is None:
            raise Exception("No likelihood given.")
        
        prior_matrix_list = []
        for idx, prior in tqdm(enumerate(self.priors), total=len(self.priors), 
                               desc='Setting up prior matrices', ncols=80):
            prior_matrices = []
            

            if not(self.hyperparameter_axes_tuple[idx][0] is None):
                for hyperparameter_axisval in tqdm(zip(*self.hyperparameter_axes_tuple[idx]), desc=f"Setting up matrices for the {idx+1}th prior", 
                                                   total=self.hyperparameter_axes_tuple[idx][0].shape[0],leave=False, ncols=90):
                    priorvals= xp.squeeze(prior.construct_prior_array(hyperparameters=(hyperparameter_axisval,), normalise=True))
                    prior_matrices.append(priorvals)
                prior_matrix_list.append(prior_matrices)
            else:
                priorvals= xp.squeeze(prior.construct_prior_array(normalise=True))
                prior_matrices.append(priorvals)
                prior_matrix_list.append(prior_matrices)

        log_marg_partial = functools.partial(self.observation_marg, prior_matrix_list=prior_matrix_list, )
        print("Starting nuisance parameter marginalisation")
        if parallelize:
            print("Sorry no progress bar for this step yet.")
            with Pool(self.numcores) as pool:
                log_margresults = pool.map(log_marg_partial, zip(*axisvals))
        else:

            log_margresults = [log_marg_partial(axisval) for axisval in tqdm(zip(*axisvals), total=len(list(axisvals[0])), ncols=80)]
        print("Nuisance parameter marginalisation finished")

        self.log_margresults = log_margresults
        
        return log_margresults
    
    
    
    def add_results(self, new_log_marg_results):
            self.log_margresults = xp.append(self.log_margresults, new_log_marg_results, axis=0)


    # Todo: Create separate mixture model class
    def create_mixture_log_posterior(self, mixture_axes=None, log_margresults=None):
        if mixture_axes is None:
            mixture_axes = self.mixture_axes
            if mixture_axes is None:
                raise Exception("Mixture axes not specified")
        else:
            self.mixture_axes = mixture_axes
        if not(self.priors is None):
            if len(mixture_axes)!=len(self.priors):
                raise Exception(f""""Number of mixture axes does not match number of components. Please check your inputs.
    Number of mixture axes is {len(mixture_axes)}
    and number of prior components is {len(self.priors)}.""")
        if log_margresults is None:
            log_margresults = self.log_margresults

        # Relative weights stay the same but now they will be 
        mixture_axes = mixture_axes/xp.sum(mixture_axes, axis=0)

        # reshape log_margresults into shape of (num_components, ...)
        reshaped_log_margresults = xp.asarray(self.log_margresults).T

        # Creating components of mixture for each prior
        mixture_array_list = []
        for idx, mixture_array in enumerate(mixture_axes):
            log_margresults_for_idx = xp.vstack(reshaped_log_margresults[idx,:])
            print(idx, xp.log(mixture_array).shape, log_margresults_for_idx.shape)
            mixture_array_list.append(xp.expand_dims(xp.log(mixture_array), axis=(*(xp.arange(log_margresults_for_idx.ndim)+1),))+log_margresults_for_idx[xp.newaxis, ...])


        # Now to get around the fact that every component of the mixture can be a different shape
        # This should be the final output except for the dimension over which there are different observations (shown here as len(self.log_margresults))
        final_output_shape = [*[mixture_array.shape for mixture_array in mixture_axes], len(self.log_margresults)]
        final_output_shape.pop(0)

        # Axes for each of the priors to __not__ expand in
        prioraxes = []
        # Counter for how many hyperparameter axes we have used
        hyper_idx = 0
        for idx, hyperparameteraxes in enumerate(self.hyperparameter_axes_tuple):
            prior_axis_instance = list(xp.arange(len(mixture_axes)))
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

            mixture_array =xp.expand_dims(mixture_array, axis=axis)

            mixture_array_list[prior_idx] = mixture_array


        # Now to combine the results for all the data and to 
        combined_mixture = -xp.inf
        for mixture_component in mixture_array_list:
            combined_mixture = xp.logaddexp(combined_mixture, mixture_component)

        unnormed_log_posterior = xp.sum(combined_mixture, axis=1)

        # # Saving the result to the object
        # self.unnormed_log_posterior = unnormed_log_posterior

        return unnormed_log_posterior
    
    def combine_results(self, unnormed_log_posterior):
        self.unnormed_log_posterior += unnormed_log_posterior


    def apply_hyperparameter_prior_values(self, hyperparameter_prob_mesh):
        if self.priors_applied:
            self.unnormed_log_posterior = self.unnormed_log_posterior+hyperparameter_prob_mesh
            self.priors_applied = 1
        else:
            warnings.warn('Hyperparameter priors have already been applied')






            
            
    def save_data(self, directory_path='', reduce_data_consumption=True, save_log_margresults=True):
        
        data_to_save = {}
        
        data_to_save['hyperparameter_axes_tuple']   = self.hyperparameter_axes_tuple
        if not(reduce_data_consumption):
            data_to_save['priors']                  = self.priors
            data_to_save['likelihood']              = self.likelihood
        if save_log_margresults:
            data_to_save['log_margresults']             = self.log_margresults

            
        data_to_save['dependent_logjacob']          = self.dependent_logjacob
        data_to_save['axes']                        = self.axes
        data_to_save['dependent_axes']              = self.dependent_axes
        data_to_save['mixture_axes']                = self.mixture_axes
        data_to_save['unnormed_log_posterior']      = self.unnormed_log_posterior
        data_to_save['priors_applied']              = self.priors_applied
        
        
        
        
        save_file_path = os.path.join(directory_path, "hyper_parameter_data.pkl")
        
        
        try:
            with open(save_file_path, 'wb') as pickle_file:
                pickle.dump(data_to_save, pickle_file)

            print("Data saved to", save_file_path)
        except:
            warnings.warn("Something went wrong when trying to save to the specified directory. Saving to current working directory")
            try:
                with open("hyper_parameter_data.pkl", 'wb') as pickle_file:
                        pickle.dump(data_to_save, pickle_file)

                print("Data saved to working directory")
            except:
                raise Exception("Attempts to save result failed.")
            
    def plot_posterior(self, identifier=None, **kwargs):
        if identifier is None:
            if not(self.name==''):
                identifier=self.name
            

        plot_posterior(log_posterior= xp.squeeze(self.unnormed_log_posterior - logsumexp(self.unnormed_log_posterior)),
                       xi_range=self.mixture_axes[0], logmassrange=self.hyperparameter_axes_tuple[0][0],
                       identifier=identifier, **kwargs)

        