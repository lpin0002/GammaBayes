from scipy.special import logsumexp
from functools import partial
import time
try:
    import jax
    import jax.numpy as np

    from jax import random
    from jax.lax import cumlogsumexp
    from .sampler_utils import RandomGen


    random_gen_class = RandomGen()


    logaddexp_accumulate = cumlogsumexp

    random_gen = random_gen_class.random




except:
    import numpy as np
    from numpy.random import random as random_gen
    from numpy import logaddexp
    logaddexp_accumulate = logaddexp.accumulate

import random
from numpy import ndarray


def inverse_transform_sampler(logpmf, Nsamples=1):
    """Function to perform inverse transform sampling on the input discrete log
        probability density values

    Args:
        logpmf (ndarray): discrete log probability density values
        Nsamples (int, optional): Number of wanted samples. Defaults to 1.

    Returns:
        ndarray: Sampled indices of the input axis
    """
    logpmf = logpmf - logsumexp(logpmf)
    logcdf = logaddexp_accumulate(logpmf)
    cdf = np.exp(logcdf-logcdf[-1])  

    randvals = [random_gen() for xkcd in range(Nsamples)]
    indices = np.asarray([np.searchsorted(cdf, u) for u in randvals])
    return indices

# Need to figure out a more rigorous solution. Stable for up to ...
def integral_inverse_transform_sampler(logpmf, axes=None, Nsamples: int = 1):

    Nsamples = int(round(Nsamples))

    logpmf_flattened = logpmf.flatten()

    logpmf_flattened = logpmf_flattened - logsumexp(logpmf_flattened)

    flat_logcdf = logaddexp_accumulate(logpmf_flattened)

    flat_cdf = np.exp(flat_logcdf-flat_logcdf[-1])  

    randvals = [random_gen() for xkcd in range(Nsamples)]
    indices = np.asarray([np.searchsorted(flat_cdf, u) for u in randvals])

    reshaped_simulated_indices = np.unravel_index(indices, np.squeeze(logpmf).shape)

    return [axis[index] for axis, index in zip(axes, reshaped_simulated_indices)]




def vectorised_inverse_transform_sampler(logpmfs, axes=None, Nsamples: list[int] = [1], reshape_shape=None):

    Nsamples = [int(round(num)) for num in Nsamples]

    logpmfs_flattened = logpmfs.reshape(logpmfs.shape[0], -1)
    logpmfs_flattened = logpmfs_flattened - logsumexp(logpmfs_flattened, axis=1)[:, None]
    flat_logcdfs = logaddexp_accumulate(logpmfs_flattened, axis=1)
    flat_cdfs = np.exp(flat_logcdfs-flat_logcdfs[:, -1][:, None])
    # Vectorized random value generation, not sure whether it's any better than list comprehension
    randvals_list = random_gen((len(Nsamples), max(Nsamples)))

    indices_list = [np.searchsorted(flat_cdf, randvals[:num]) for flat_cdf, randvals, num in zip(flat_cdfs, randvals_list, Nsamples)]

    samples  = [[] for axis in axes]

    for indices in indices_list:

        reshaped_simulated_indices = np.unravel_index(indices, np.squeeze(logpmfs[0, :]).shape)

        [samples[axis_index].extend(axis[index]) for axis_index, (axis, index) in enumerate(zip(axes, reshaped_simulated_indices))]

    return samples
    