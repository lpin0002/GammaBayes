import numpy as np
from scipy.special import logsumexp
import functools
from tqdm.auto import tqdm
from gammabayes.utils import angularseparation, convertlonlat_to_offset
from multiprocessing.pool import ThreadPool as Pool


class hyperparameter_likelihood(object):
    def __init__(self, priors, likelihood, axes=None, dependent_axes=None, dependent_logjacob=0, hyperparameter_axes=(), numcores=8,likelihoodnormalisation= ()):
        self.priors = priors
        self.likelihood = likelihood
        
        self.axes = axes
            
        if dependent_axes is None:
            self.dependent_axes = self.likelihood[0].dependent_axes
        else:
            self.dependent_axes = dependent_axes
            
        self.dependent_logjacob = dependent_logjacob
        self.hyperparameter_axes = hyperparameter_axes
        self.numcores = numcores
        self.likelihoodnormalisation = likelihoodnormalisation
        
        
    
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
            

            
        return np.array([np.squeeze(np.array(sigmargresults)), bkgmargresult])


    def full_obs_marginalisation(self, axisvals):

        
        # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
        np.seterr(divide='ignore', invalid='ignore')
        # log10eaxistrue,  longitudeaxistrue, latitudeaxistrue = axes
                
        nuisance_loge_setup_mesh, nuisance_longitude_setup_mesh, nuisance_latitude_setup_mesh = np.meshgrid(*self.dependent_axes, indexing='ij')
        

        prior_matrix_list = []
        for idx, prior in tqdm(enumerate(self.priors), total=len(self.priors), desc='Setting up prior matrices'):
            prior_matrices = []
            for hyperparametervalue in self.hyperparameter_axes[idx]:
                priorvals= np.squeeze(prior.construct_prior_array(hyperparameters=(hyperparametervalue,), normalise=True))
                prior_matrices.append(priorvals)
            prior_matrix_list.append(prior_matrices)
            

        marg_partial = functools.partial(self.observation_marg, signal_prior_matrices=prior_matrix_list[0], logbkgpriorvalues=prior_matrix_list[1])

        with Pool(self.numcores) as pool:
                    margresults = pool.map(marg_partial, tqdm(zip(*axisvals), total=len(list(axisvals[0])), desc='Performing parallelized direct event marginalisation'))

                    
        return margresults