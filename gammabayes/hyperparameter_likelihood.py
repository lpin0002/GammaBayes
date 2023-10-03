import numpy as np
from scipy.special import logsumexp
import functools
from tqdm.auto import tqdm
from .utils.utils import angularseparation, convertlonlat_to_offset
from multiprocessing.pool import ThreadPool as Pool
import json, os, warnings

class hyperparameter_likelihood(object):
    def __init__(self, priors, likelihood, axes=None,
                 dependent_axes=None, dependent_logjacob=0, 
                 hyperparameter_axes=(), numcores=8, 
                 likelihoodnormalisation= (), marg_results=None, mixture_axes = None):
        
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
        self.marg_results               = marg_results
        if mixture_axes is None:
            self.mixture_axes               = np.array([np.linspace(0,1,101)]*len(priors))
        if len(mixture_axes) != len(priors):
            self.mixture_axes               = np.array([mixture_axes]*len(priors))
        else:
            self.mixture_axes = np.array([*mixture_axes])
        
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
        for idx, prior in tqdm(enumerate(self.priors), total=len(self.priors), 
                               desc='Setting up prior matrices', ncols=80):
            prior_matrices = []
            for hyperparametervalue in self.hyperparameter_axes[idx]:
                priorvals= np.squeeze(prior.construct_prior_array(hyperparameters=(hyperparametervalue,), normalise=True))
                prior_matrices.append(priorvals)
            prior_matrix_list.append(prior_matrices)
            

        marg_partial = functools.partial(self.observation_marg, signal_prior_matrices=prior_matrix_list[0], logbkgpriorvalues=prior_matrix_list[1])

        with Pool(self.numcores) as pool:
                    margresults = pool.map(marg_partial, tqdm(zip(*axisvals), total=len(list(axisvals[0])), desc='Performing parallelized direct event marginalisation'))

        
        
        
        return margresults
    
    
    
    def add_results(self, new_marg_results):
            self.marg_results = np.append(self.marg_results, new_marg_results, axis=0)
            
    def mixture(self, mixture_axes):
        if mixture_axes is None:
            mixture_axes = self.mixture_axes
        else:
            self.mixture_axes = mixture_axes
            
        pass
            
            
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
        