from gammabayes.hyper_inference.discrete_brute_scan_hyperparameter_likelihood import discrete_brute_scan_hyperparameter_likelihood
from gammabayes.utils import iterate_logspace_integration, logspace_riemann, update_with_defaults
from gammabayes.utils.event_axes import derive_edisp_bounds, derive_psf_bounds
from gammabayes.utils.config_utils import save_config_file
from gammabayes import EventData

import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import os, warnings, logging, time, functools, numpy as np


class discrete_adaptive_scan_hyperparameter_likelihood(discrete_brute_scan_hyperparameter_likelihood):
    def __init__(self, *args, 
                 bounds: list[ str, float] = None,
                 bounding_percentiles: list[float]      = [90 ,90],
                 bounding_sigmas: list[int]     = [4,4],
                 **kwargs):
        # Initialisation of the base class discrete_brute_scan_hyperparameter_likelihood
        super().__init__(*args, **kwargs)

        self.bounds = bounds

        # If the bounds aren't given then they are calculated based on the input log likelihood
        if self.bounds is None:
            _, logeval_bound               = derive_edisp_bounds(irf_loglike=self.likelihood, percentile=bounding_percentiles[0], sigmalevel=bounding_sigmas[0])
            lonlact_bound                   = derive_psf_bounds(irf_loglike=self.likelihood, percentile=bounding_percentiles[1], sigmalevel=bounding_sigmas[1], 
                                                                axis_buffer=1, parameter_buffer = np.squeeze(np.ptp(self.dependent_axes[1]))/2)
            self.bounds = [['log10', logeval_bound], ['linear', lonlact_bound], ['linear', lonlact_bound]]

        logging.debug(f"Bounds: {self.bounds}")

    def bound_axis(self, 
                   axis: np.ndarray, 
                   bound_type: str, 
                   bound_radii: float, 
                   estimated_val: float):
        if bound_type=='linear':
            axis_indices = np.where(
            (axis>estimated_val-bound_radii) & (axis<estimated_val+bound_radii) )[0]

        elif bound_type=='log10':
            axis_indices = np.where(
            (np.log10(axis)>np.log10(estimated_val)-bound_radii) & (np.log10(axis)<np.log10(estimated_val)+bound_radii) )[0]
            
        temp_axis = axis[axis_indices]

        return temp_axis, axis_indices
    

    def observation_nuisance_marg(self, 
                                  event_vals: np.ndarray | EventData, 
                                  prior_matrix_list: list[np.ndarray]):

        temp_axes_and_indices = [self.bound_axis(axis, bound_type=bound_info[0], bound_radii=bound_info[1], estimated_val=event_val) for bound_info, axis, event_val in zip(self.bounds, self.likelihood.dependent_axes, event_vals)]
        temp_axes = [temp_axis_info[0] for temp_axis_info in temp_axes_and_indices]

        meshvalues  = np.meshgrid(*event_vals, *temp_axes, indexing='ij')
        index_meshes = np.ix_( *[temp_axis_info[1] for temp_axis_info in temp_axes_and_indices])

        flattened_meshvalues = [meshmatrix.flatten() for meshmatrix in meshvalues]
        
        likelihoodvalues = np.squeeze(self.likelihood(*flattened_meshvalues).reshape(meshvalues[0].shape))

        likelihoodvalues = likelihoodvalues - self.likelihoodnormalisation[*index_meshes]
        
        all_log_marg_results = []

        for prior_matrices in prior_matrix_list:
            # Transpose is because of the convention for where I place the 
                # axes of the true values (i.e. first three indices of the prior matrices)
                # and numpy doesn't support this kind of matrix addition nicely
            logintegrandvalues = (np.squeeze(prior_matrices).T[...,index_meshes[2].T,index_meshes[1].T,index_meshes[0].T]+np.squeeze(likelihoodvalues).T).T
            
            
            single_parameter_log_margvals = iterate_logspace_integration(logintegrandvalues,   
                axes=temp_axes, axisindices=[0,1,2])

            all_log_marg_results.append(np.squeeze(single_parameter_log_margvals))

        
        return np.array(all_log_marg_results, dtype=object)


    def pack_data(self, reduce_mem_consumption: bool = True):

        packed_data = self._pack_data(reduce_mem_consumption)

        # Only 'new' information in this daughter class are 
            # the bounds used for the restricted parameter range
            # marginalisation
        packed_data['bounds'] = self.bounds

        return packed_data
