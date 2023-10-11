from scipy.special import logsumexp
import numpy as np
import random


def inverse_transform_sampling(logpmf, Nsamples=1):
    
    logpmf = logpmf - logsumexp(logpmf)
    logcdf = np.logaddexp.accumulate(logpmf)
    cdf = np.exp(logcdf-logcdf[-1])  

    randvals = [random.random() for xkcd in range(Nsamples)]
    indices = [np.searchsorted(cdf, u) for u in randvals]
    return indices
    
    