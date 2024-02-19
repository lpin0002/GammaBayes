import numpy as np

def bound_axis(axis: np.ndarray, 
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


def default_proposal_prior_array(axes):
    return np.meshgrid(np.log(1+0*axes[0]), axes[1], axes[2], indexing='ij')[0]


def discrete_prior_transform(u, inv_cdf_func=None, log_prior_array=None, axes=None):
        output_index = int(np.round(inv_cdf_func(u[0])))
        reshaped_indices = np.unravel_index(output_index, shape=log_prior_array.shape)
        output = [axis[output_idx] for output_idx, axis in zip(reshaped_indices, axes)]
        return output


class ResultsWrapper:
    def __init__(self, results_dict):
        self.__dict__.update(results_dict)
    
    def __getattr__(self, attr):
        # This method is called if the attribute wasn't found the usual ways
        raise AttributeError(f"'ResultsWrapper' object has no attribute '{attr}'")

