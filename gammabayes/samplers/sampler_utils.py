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


from scipy import special
from scipy.interpolate import interp1d
from tqdm import tqdm
import numpy as np
import functools, dynesty, warnings, os, sys, time
from matplotlib import pyplot as plt
from gammabayes.utils.event_axes import derive_edisp_bounds, derive_psf_bounds
from gammabayes.utils import update_with_defaults, iterate_logspace_integration


class dynesty_restricted_proposal_marg_wrapper(object):

    def __init__(self, measured_event, proposal_prior, irf_loglike, 
                 nuisance_axes, 
                 logenergy_bound=0.5, lonlat_bound=0.5):
        
        self.measured_event = measured_event

        self.restricted_energy = nuisance_axes[0][
                np.where(
                    (nuisance_axes[0]>10**(np.log10(measured_event[0])-logenergy_bound)) & (nuisance_axes[0]<10**(np.log10(measured_event[0])+logenergy_bound))
                    )
                    ]

        self.restricted_longitude = nuisance_axes[1][
                np.where(
                    (nuisance_axes[1]>measured_event[1]-lonlat_bound) & (nuisance_axes[1]<measured_event[1]+lonlat_bound)
                    )
                    ]
        self.restricted_latitude = nuisance_axes[2][
                    np.where(
                        (nuisance_axes[2]>measured_event[2]-lonlat_bound) & (nuisance_axes[2]<measured_event[2]+lonlat_bound)
                        )
                        ]
        
        nuisance_mesh = np.meshgrid(*measured_event, 
                                    self.restricted_energy,  
                                    self.restricted_longitude, 
                                    self.restricted_latitude, indexing='ij')
        
        self.prior_matrix = np.squeeze(proposal_prior(*nuisance_mesh[3:]))
        
        logpdf = self.prior_matrix
        self.logpdf_shape = logpdf.shape
        flattened_logpdf = logpdf.flatten()
        flattened_logcdf = np.logaddexp.accumulate(flattened_logpdf)
        flattened_logcdf = flattened_logcdf


        self.pseudo_log_norm = flattened_logcdf[-1]
        if not(np.isinf(self.pseudo_log_norm)):
            flattened_logcdf = flattened_logcdf - self.pseudo_log_norm

        self.flattened_cdf = np.exp(flattened_logcdf)

        self.interpolation_func = interp1d(y=np.arange(flattened_logcdf.size), x=self.flattened_cdf, 
                                    fill_value=(0, flattened_logcdf.size-1), bounds_error=False)
        

        self.log_likelihood = functools.partial(
            irf_loglike.irf_loglikelihood.dynesty_single_loglikelihood,
            recon_energy=measured_event[0], recon_lon=measured_event[1], recon_lat=measured_event[2]
        )



    def prior_transform(self, u):
        flat_idx = self.interpolation_func(u[0])
        flat_idx = int(np.round(flat_idx))
        axis_indices = np.unravel_index(shape=self.logpdf_shape, indices=flat_idx)
        axis_values = [self.restricted_energy[axis_indices[0]], self.restricted_longitude[axis_indices[1]], self.restricted_latitude[axis_indices[2]]]
        return axis_values
    
    def likelihood(self, x):
        return self.log_likelihood(x)