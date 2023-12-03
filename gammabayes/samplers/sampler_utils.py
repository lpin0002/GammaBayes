import numpy as np
from gammabayes.utils.event_axes import energy_true_axis, longitudeaxistrue, latitudeaxistrue

def construct_constrained_axes(measured, axes=[energy_true_axis, longitudeaxistrue, latitudeaxistrue], num_sigmas=8, axes_sigmas=[0.05,0.07,0.07]):
    constructed_axis1 = axes[0][np.where(
        np.logical_and(
            np.log10(axes[0])>=np.log10(measured[0])-num_sigmas*axes_sigmas[0], 
            np.log10(axes[0])<=np.log10(measured[0])+num_sigmas*axes_sigmas[0]
            ))]
    constructed_axis2 = axes[1][np.where(
        np.logical_and(
            axes[1]>=measured[1]-num_sigmas*axes_sigmas[1], 
            axes[1]<=measured[1]+num_sigmas*axes_sigmas[1])
            )]
    constructed_axis3 = axes[2][np.where(
        np.logical_and(
            axes[2]>=measured[2]-num_sigmas*axes_sigmas[2], 
            axes[2]<=measured[2]+num_sigmas*axes_sigmas[2])
            )]
    
    return [constructed_axis1, constructed_axis2, constructed_axis3]


def default_proposal_prior_array(axes):
    return np.meshgrid(np.log(1+0*axes[0]), axes[1], axes[2], indexing='ij')[0]


def discrete_prior_transform(u, inv_cdf_func=None, log_prior_array=None, axes=None):
        output_index = int(np.round(inv_cdf_func(u[0])))
        reshaped_indices = np.unravel_index(output_index, shape=log_prior_array.shape)
        output = [axis[output_idx] for output_idx, axis in zip(reshaped_indices, axes)]
        return output