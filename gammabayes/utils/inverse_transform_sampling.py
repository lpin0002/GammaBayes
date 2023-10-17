from scipy.special import logsumexp
import numpy as np
import random


def inverse_transform_sampling(logpmf, Nsamples=1):
    """Function to perform inverse transform sampling on the input discrete log
        probability density values

    Args:
        logpmf (np.ndarray): discrete log probability density values
        Nsamples (int, optional): Number of wanted samples. Defaults to 1.

    Returns:
        np.ndarray: Sampled indices of the input axis
    """
    
    logpmf = logpmf - logsumexp(logpmf)
    logcdf = np.logaddexp.accumulate(logpmf)
    cdf = np.exp(logcdf-logcdf[-1])  

    randvals = [random.random() for xkcd in range(Nsamples)]
    indices = [np.searchsorted(cdf, u) for u in randvals]
    return indices
    
    