from scipy.special import logsumexp
import numpy as np
import random


def inverse_transform_sampler(logpmf, Nsamples=1):
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

# Need to figure out a more rigorous solution. Stable for up to ...
def integral_inverse_transform_sampler(logpmf, axes=None, log10unif_axes = [], Nsamples=1, logjacob=None):

    if logjacob is None:
        if axes is None:
            raise Exception("No axes or jacobian given.")
        else:
            # Following bit of code is to make it so the jacobian matrix is the same 
            #   size as the logpmf matrix
            jacobian_axes = [makelogjacob(axis) for axis in axes[log10unif_axes]]
            uniform_axes_and_logjacob_axes = []
            counter_idx = 0
            for idx in range(len(axes)):
                if idx in log10unif_axes:
                    uniform_axes_and_logjacob_axes.append(jacobian_axes[counter_idx])
                    counter_idx+=1
                else:
                    uniform_axes_and_logjacob_axes.append(axes[idx])
            
            logjacob = np.sum(np.meshgrid(*uniform_axes_and_logjacob_axes, 
                                                    indexing='ij')[log10unif_axes])

    logpmf_with_jacob = logpmf+logjacob

    logpmf_with_jacob_flattened = logpmf_with_jacob.flatten()

    logpmf_with_jacob_flattened = logpmf_with_jacob_flattened - logsumexp(logpmf_with_jacob_flattened)

    flat_logcdf = np.logaddexp.accumulate(logpmf_with_jacob_flattened)

    flat_cdf = np.exp(flat_logcdf-flat_logcdf[-1])  

    randvals = [random.random() for xkcd in range(Nsamples)]
    indices = [np.searchsorted(flat_cdf, u) for u in randvals]

    reshaped_simulated_indices = np.unravel_index(indices, np.squeeze(logpmf).shape)

    return [axis[index] for axis, index in zip(axes, reshaped_simulated_indices)]
    