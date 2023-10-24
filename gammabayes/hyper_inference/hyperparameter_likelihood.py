import numpy as np
from scipy.special import logsumexp
import functools
from tqdm.auto import tqdm
from gammabayes.utils.utils import angularseparation, convertlonlat_to_offset
from multiprocessing.pool import ThreadPool as Pool
import json, os, warnings

class hyperparameter_likelihood(object):
    def __init__(self, priors, likelihood, axes=None,
                 dependent_axes=None, dependent_logjacob=0, 
                 hyperparameter_axes=(), numcores=8, 
                 likelihoodnormalisation= (), log_margresults=None, mixture_axes = None,
                 unnormalised_log_posterior=0):
        
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

        self.unnormalised_log_posterior = unnormalised_log_posterior
        
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


    
    def observation_marg(self, axisvals, signal_prior_matrices, logbkgpriorvalues):
        # psfnormvalues, edispnormvalues = self.likelihoodnormalisation
        sigmargresults = []
        
        meshvalues  = np.meshgrid(*axisvals, *self.dependent_axes, indexing='ij')
        flattened_meshvalues = [meshmatrix.flatten() for meshmatrix in meshvalues]
        
        
        likelihoodvalues = self.likelihood(*flattened_meshvalues).reshape(meshvalues[0].shape)
        
        likelihoodvalues = likelihoodvalues - self.likelihoodnormalisation
        
        bkgmargresult = logsumexp(logbkgpriorvalues+self.dependent_logjacob+likelihoodvalues)

        
        singlemass_sigmargvals = []
        for logmass_idx, signal_prior_matrix in enumerate(signal_prior_matrices):
            
            output = logsumexp(signal_prior_matrix+self.dependent_logjacob+likelihoodvalues)
            singlemass_sigmargvals.append(output)

        sigmargresults.append(singlemass_sigmargvals)
            
            
        return np.array([np.squeeze(np.array(sigmargresults)), bkgmargresult], dtype=object)


    def full_obs_marginalisation(self, axisvals):

        
        # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
        np.seterr(divide='ignore', invalid='ignore')
        # log10eaxistrue,  longitudeaxistrue, latitudeaxistrue = axes
                
        nuisance_loge_setup_mesh, nuisance_longitude_setup_mesh, nuisance_latitude_setup_mesh = np.meshgrid(*self.dependent_axes, indexing='ij')
        

        prior_matrix_list = []
        for idx, prior in tqdm(enumerate(self.priors), total=len(self.priors), desc='Setting up prior matrices'):
            prior_matrices = []
            for hyperparametervalue in zip(*np.meshgrid(*self.hyperparameter_axes[idx])):
                priorvals= np.squeeze(prior.construct_prior_array(hyperparameters=(hyperparametervalue,), normalise=True))
                prior_matrices.append(priorvals)
            prior_matrix_list.append(prior_matrices)
            

        marg_partial = functools.partial(self.observation_marg, signal_prior_matrices=prior_matrix_list[0], logbkgpriorvalues=prior_matrix_list[1])

        with Pool(self.numcores) as pool:
                    margresults = pool.map(marg_partial, tqdm(zip(*axisvals), total=len(list(axisvals[0])), desc='Performing parallelized direct event marginalisation'))

        
        
        
        return margresults
    
    
    
    def add_results(self, new_log_marg_results):
            self.log_margresults = np.append(self.log_margresults, new_log_marg_results, axis=0)
            
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
        mixture_axes = mixture_axes/np.sum(mixture_axes, axis=0)

        # reshape log_margresults into shape of (num_components, ...)
        reshaped_log_margresults = np.asarray(log_margresults).T
        print(f"Reshaped mixture shape: {reshaped_log_margresults.shape}")
        print(f"reshaped_log_margresults[0,:]: {reshaped_log_margresults[0,:]}")

        # Creating components of mixture for each prior
        mixture_array_list = []
        for idx, mixture_array in enumerate(mixture_axes):
            log_margresults_for_idx = np.vstack(reshaped_log_margresults[idx,:])
            print(idx, np.log(mixture_array).shape, log_margresults_for_idx.shape)
            mixture_array_list.append(np.expand_dims(np.log(mixture_array), axis=(*(np.arange(log_margresults_for_idx.ndim)+1),))+log_margresults_for_idx[np.newaxis, ...])


        # Now to get around the fact that every component of the mixture can be a different shape
        # This should be the final output except for the dimension over which there are different observations (shown here as len(self.log_margresults))
        final_output_shape = [*[mixture_array.shape for mixture_array in mixture_axes], len(log_margresults)]
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

            mixture_array =np.expand_dims(mixture_array, axis=axis)

            mixture_array_list[prior_idx] = mixture_array


        # Now to combine the results for all the data and to 
        combined_mixture = -np.inf
        for mixture_component in mixture_array_list:
            combined_mixture = np.logaddexp(combined_mixture, mixture_component)

        unnormed_log_posterior = np.sum(combined_mixture, axis=1)

        # # Saving the result to the object
        # self.unnormed_log_posterior = unnormed_log_posterior

        return unnormed_log_posterior

    def combine_posteriors(self, unnormalised_log_posterior):
        self.unnormalised_log_posterior += unnormalised_log_posterior
            
            
    def save_data(self, directory_path='', reduce_data_consumption=True):
        
        data_to_save = {}
        
        data_to_save['hyperparameter_axes']     = self.hyperparameter_axes
        if not(reduce_data_consumption):
            data_to_save['priors']              = self.priors
            data_to_save['likelihood']          = self.likelihood
            
        data_to_save['dependent_logjacob']      = self.dependent_logjacob
        data_to_save['axes']                    = self.axes
        data_to_save['dependent_axes']          = self.dependent_axes
        data_to_save['marg_results']            = self.marg_results
        data_to_save['mixture_axes']            = self.mixture_axes
        data_to_save['posterior']               = self.posterior
        
        
        serialized_data = json.dumps(data_to_save, indent=4)
        
        save_file_path = os.path.join(directory_path, "hyper_parameter_data.json")
        
        
        try:
            with open(save_file_path, "w") as file:
                file.write(serialized_data)
        except:
            warnings.warn("Something went wrong when trying to save to the specified directory. Saving to current working directory")
            with open("hyper_parameter_data.json", "w") as file:
                file.write(serialized_data)
        