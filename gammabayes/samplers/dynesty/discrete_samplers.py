import numpy as np
import dynesty
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from scipy.special import logsumexp
from scipy.interpolate import interp1d
import functools
from multiprocessing import Pool
from gammabayes.likelihoods.irfs import single_loglikelihood





def _constrain_indices(measured, marginalisation_axis,marginalisation_axis_sigma, numsigmas,):

    constrained_indices = np.where(
        np.logical_and(
            marginalisation_axis>=measured-numsigmas*marginalisation_axis_sigma, 
            marginalisation_axis<=measured+numsigmas*marginalisation_axis_sigma
            )
            )

    return constrained_indices


def single_loglikelihood_wrapper(truevals, measured):
    return single_loglikelihood(np.log10(measured[0]), measured[1], measured[2],
            np.log10(truevals[0]), truevals[1], truevals[2])[0]

def _construct_constrained_axes(measured, marginalisation_axes,marginalisation_axes_sigmas, numsigmas):
        
    energy_constrained_axis = marginalisation_axes[0][_constrain_indices(
            np.log10(measured[0]),
            np.log10(marginalisation_axes[0]),
            marginalisation_axes_sigmas[0], numsigmas)]

    lon_constrained_axis = marginalisation_axes[1][_constrain_indices(
            measured[1],
            marginalisation_axes[1],
            marginalisation_axes_sigmas[1], numsigmas)]

    lat_constrained_axis = marginalisation_axes[2][_constrain_indices(
            measured[2],
            marginalisation_axes[2],
            marginalisation_axes_sigmas[2], numsigmas)]

    constrained_axes = [energy_constrained_axis, lon_constrained_axis, lat_constrained_axis]

    return constrained_axes



def _construct_flat_prior_inv_cdf(axes):

    logpriorarray = np.meshgrid(*axes, indexing='ij')[0]*0#+np.meshgrid(np.log(axis1), axis2, axis3, indexing='ij')[0]

    flattened_logpriorarray = logpriorarray.flatten()
    logcdfarray = np.logaddexp.accumulate(flattened_logpriorarray)
    cdfarray = np.exp(logcdfarray-logcdfarray[-1])

    indices = np.arange(len(flattened_logpriorarray))
    inv_cdf_func = interp1d(x=cdfarray, y = indices, bounds_error=False, fill_value=(indices[0],indices[-1]), kind='nearest')

    return inv_cdf_func, logpriorarray

def prior_transform(u, axes, inv_cdf_func, logpriorarray):
    output_index = int(np.round(inv_cdf_func(u[0])))
    reshaped_indices = np.unravel_index(output_index, shape=logpriorarray.shape)
    output = [axis[output_idx] for output_idx, axis in zip(reshaped_indices, axes)]
    return output


class discrete_parameter_proposal_sampler:

    def __init__(self, loglike, marginalisation_axes, marginalisation_axes_sigmas, numsigmas = 8, 
        livepoints=250, numcores=1):
        self.loglike                        = loglike
        self.numsigmas                      = numsigmas
        self.livepoints                     = livepoints
        self.ndim                           = 3
        self.marginalisation_axes           = marginalisation_axes
        self.marginalisation_axes_sigmas    = marginalisation_axes_sigmas
        self.numcores                       = numcores

    # reconloge, recon_lon, recon_lat
    




    def run_dynesty(self, measured, dlogz=0.05):

        constrained_axes = _construct_constrained_axes(measured, self.marginalisation_axes, self.marginalisation_axes_sigmas, self.numsigmas)

        inv_cdf_func, logpriorarray = _construct_flat_prior_inv_cdf(constrained_axes)

        dynesty_prior_transform = functools.partial(prior_transform, axes=constrained_axes, 
            inv_cdf_func=inv_cdf_func, logpriorarray=logpriorarray)
        
        dynesty_loglike = functools.partial(single_loglikelihood_wrapper, measured=measured)

        if self.numcores>1:
            with Pool(self.numcores) as pool:
                sampler = NestedSampler(dynesty_loglike,
                    dynesty_prior_transform, self.ndim, self.livepoints, queue_size=self.numcores, pool=pool)

                sampler.run_nested(dlogz=dlogz)
        else:
            sampler = NestedSampler(loglike_func,
                    dynesty_prior_transform, self.ndim, self.livepoints)

            sampler.run_nested(dlogz=dlogz)


        results = sampler.results
        self.results = results

        return results