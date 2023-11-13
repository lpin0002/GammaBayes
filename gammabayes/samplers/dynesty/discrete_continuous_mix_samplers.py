import dynesty
import numpy as np
import warnings
from multiprocessing import Pool
class discrete_hyperparameter_continuous_mix_post_process_sampler(object):
    def __init__(self, hyper_param_ranges_tuple=None, mixture_axes=None, margresultsarray=None, numcores=1, multiprocess=False, **kwargs):
        self.hyper_param_ranges_tuple                   = hyper_param_ranges_tuple
        flattened_hyper_param_ranges                    = []
        num_discrete_nontrivial_hyperparameters         = 0

        for hyper_param_ranges in hyper_param_ranges_tuple:
            if type(hyper_param_ranges[0])!=type(None):
                for hyper_param_range in hyper_param_ranges:
                    num_discrete_nontrivial_hyperparameters+=1
                    flattened_hyper_param_ranges.append(hyper_param_range)

        self.num_discrete_nontrivial_hyperparameters    = num_discrete_nontrivial_hyperparameters

        self.flattened_hyper_param_ranges   = flattened_hyper_param_ranges
        self.mixture_axes                   = mixture_axes
        self.margresultsarray               = margresultsarray
        self.__dict__.update(kwargs)
        num_discrete_hyperparameters        = len(flattened_hyper_param_ranges)
        self.num_mixtures                   = len(mixture_axes)
        self.num_discrete_hyperparameters   = num_discrete_hyperparameters
        self.numcores                       = numcores
        self.multiprocess                   = True
        if numcores>1:
            self.multiprocess               = True
            warnings.warn("More than 1 core has been set, multiprocessing has been turned on.")




    def prior_transform(self, u):
        for _hyper_indx in range(self.num_discrete_nontrivial_hyperparameters):
            _hyperparameter_range = self.flattened_hyper_param_ranges[_hyper_indx]
            u[self.num_mixtures+_hyper_indx] = _hyperparameter_range[0]+u[self.num_mixtures+_hyper_indx]*np.ptp(_hyperparameter_range)
        return  u

    def apply_direchlet_stick_breaking_direct(self,xivals, depth):
        direchletmesh = 1

        for i in range(depth):
            direchletmesh*=(1-xivals[i])
        if depth!=len(xivals):
            direchletmesh*=xivals[depth]

        return direchletmesh


    def ln_likelihood(self, x,):

        tuple_hyperparameter_values = x[-self.num_discrete_nontrivial_hyperparameters:]
        mixture_vals = x[:-self.num_discrete_nontrivial_hyperparameters]
        logmargresults = []

        _ln_like_hyper_indices = 0
        for idx, hyperparameter_ranges in enumerate(self.hyper_param_ranges_tuple):
            zipped_tuples = np.array(list(zip(*[hyperparametermesh.flatten() for hyperparametermesh in np.meshgrid(*hyperparameter_ranges, indexing='ij')])))
            
            if type(hyperparameter_ranges[0])!=type(None):
                _hyper_param_tuple = tuple_hyperparameter_values[_ln_like_hyper_indices:_ln_like_hyper_indices+len(hyperparameter_ranges)]
                _hyper_index = np.linalg.norm(np.array(zipped_tuples)-np.array(_hyper_param_tuple), axis=1).argmin()
                
            else:
                _hyper_index = None

            tempmargresultsarray = self.margresultsarray.T[idx]

            tempmargresultsarray = np.vstack(tempmargresultsarray)
            tempmargresultsarray = np.squeeze(tempmargresultsarray[:,_hyper_index])
            logmargresults.append(tempmargresultsarray)
            if type(hyperparameter_ranges[0])!=type(None):
                _ln_like_hyper_indices+=len(hyperparameter_ranges)

        ln_like = -np.inf
        
        for idx, logmargresult in enumerate(logmargresults):
            ln_like= np.logaddexp(ln_like,
                np.squeeze(np.log(self.apply_direchlet_stick_breaking_direct(xivals=mixture_vals, depth=idx))+logmargresult))
        return np.sum(ln_like)

    def generate_log_hyperlike(self, nestedsampler_kwarg_dict={}, run_nested_kwarg_dict={}, numcores=None):
        if numcores==None:
            numcores=self.numcores
        if self.multiprocess or (numcores>1):
            warnings.warn("Calculating hyper log-likelihood with multiprocessing.")
            with Pool(self.numcores) as pool:
                sampler = dynesty.NestedSampler(self.ln_likelihood, self.prior_transform, self.num_mixtures+self.num_discrete_hyperparameters,pool=pool,
                    queue_size=self.numcores,
                    **nestedsampler_kwarg_dict) 
                sampler.run_nested(**run_nested_kwarg_dict)
                sresults = sampler.results
        else:
            sampler = dynesty.NestedSampler(self.ln_likelihood, self.prior_transform, self.num_mixtures+self.num_discrete_hyperparameters,
                **nestedsampler_kwarg_dict) 
            sampler.run_nested(**run_nested_kwarg_dict)
            sresults = sampler.results

        return sresults