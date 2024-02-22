import numpy as np, h5py

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
    
    @classmethod
    def save(cls, file_name, sampler_results, write_mode='w-'):
         
        with h5py.File(file_name, write_mode) as h5f:
            # Save samples
            if hasattr(sampler_results, 'samples'):
                h5f.create_dataset('samples', data=np.array(sampler_results.samples))
                
                # Save log weights
            if hasattr(sampler_results, 'logwt'):
                h5f.create_dataset('logwt', data=np.array(sampler_results.logwt))
            
            # Save log likelihoods
            if hasattr(sampler_results, 'logl'):
                h5f.create_dataset('logl', data=np.array(sampler_results.logl))
            
            # Save evidence information, if available
            if hasattr(sampler_results, 'logz'):
                h5f.create_dataset('logz', data=np.array(sampler_results.logz))
            
            if hasattr(sampler_results, 'logzerr'):
                h5f.create_dataset('logzerr', data=np.array(sampler_results.logzerr))

            if hasattr(sampler_results, 'information'):
                h5f.create_dataset('information', data=np.array(sampler_results.information))


            try:
                h5f.create_dataset('samples_equal', data=np.array(sampler_results.samples_equal()))
            except Exception as err:
                print(f"Could not save 'samples_equal':{err}")

            if hasattr(sampler_results, 'nlive'):
                h5f.attrs['nlive'] = int(sampler_results.nlive)

            if hasattr(sampler_results, 'niter'):
                h5f.attrs['niter'] = int(sampler_results.niter)

            if hasattr(sampler_results, 'eff'):
                h5f.attrs['eff'] = float(sampler_results.niter)

    @classmethod
    def load(cls, h5f=None, file_name=None, read_mode='r'):
        if isinstance(h5f, str):
            file_name = h5f
            h5f=None
        
        if (h5f is None):
            if file_name is None:
                raise ValueError("Either an h5py object or a file name must be provided.")
            # Open the file to get the h5py object
            h5f = h5py.File(file_name, read_mode)            


        sampler_results = {}
         
        # Load samples
        if 'samples' in h5f:
            sampler_results['samples'] = np.array(h5f['samples'])
        
        # Load log weights
        if 'logwt' in h5f:
            sampler_results['logwt'] = np.array(h5f['logwt'])
        
        # Load log likelihoods
        if 'logl' in h5f:
            sampler_results['logl'] = np.array(h5f['logl'])
        
        # Load evidence information, if available
        if 'logz' in h5f:
            sampler_results['logz'] = np.array(h5f['logz'])
        if 'logzerr' in h5f:
            sampler_results['logzerr'] = np.array(h5f['logzerr'])

        if 'information' in h5f:
            sampler_results['information'] = np.array(h5f['information'])

        if 'samples_equal' in h5f:
            sampler_results['samples_equal'] = np.array(h5f['samples_equal'])


        if 'nlive' in h5f:
            sampler_results['nlive'] = int(h5f['nlive'])

        if 'niter' in h5f:
            sampler_results['niter'] = int(h5f['niter'])

        if 'eff' in h5f:
            sampler_results['eff'] = float(h5f['eff'])

        return cls(sampler_results)