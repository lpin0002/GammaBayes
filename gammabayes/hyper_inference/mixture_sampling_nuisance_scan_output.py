import numpy as np, warnings, dynesty, logging
from gammabayes.utils import apply_direchlet_stick_breaking_direct, update_with_defaults
from gammabayes.hyper_inference.utils import _handle_parameter_specification
from gammabayes.samplers.sampler_utils import ResultsWrapper
from gammabayes import ParameterSet
import h5py



class ScanOutput_StochasticMixtureFracPosterior(object):
    """
    A class designed to handle stochastic exploration of posterior distributions for mixture fractions using the Dynesty sampler.

    Attributes:
        prior_parameter_specifications (list | dict | ParameterSet): Specifications for prior parameters.
        
        mixture_parameter_specifications (ParameterSet): Specifications for mixture parameters.
        
        log_nuisance_marg_results (np.array): Logarithm of marginal results for the parameters.
        
        log_nuisance_marg_regularisation (float): Regularisation parameter for nuisance marginalisation.
        
        num_priors (int): The number of prior distributions provided.
        
        num_events (int): The number of events in the log marginal results.
        
        mixture_bounds (list): Bounds for the mixture parameters.
        
        num_mixes (int): The number of mixture components.
        
        hyperparameter_axes (list): Axes for hyperparameters, derived from prior parameter specifications.
        
        index_to_hyper_parameter_info (dict): Mapping of hyperparameter indices to their specifications.
        
        ndim (int): The number of dimensions for the sampler, including mixture components and hyperparameters.
        
        sampler (dynesty.NestedSampler): The Dynesty sampler object used for exploration.
    """
    def __init__(self, 
                 log_nuisance_marg_results,
                 mixture_parameter_specifications:list | dict | ParameterSet,
                 log_nuisance_marg_regularisation: float = 0., # Argument for consistency between classes
                 prior_parameter_specifications: list | dict | ParameterSet = None,
                 _sampler_results={}):
        """
        Initializes the ScanOutput_StochasticMixtureFracPosterior class with necessary parameters and configurations.

        Args:
            log_nuisance_marg_results (np.array): Logarithm of marginal results for the parameters.
            
            mixture_parameter_specifications (list | dict | ParameterSet): Specifications for mixture parameters.
            
            log_nuisance_marg_regularisation (float, optional): Regularisation parameter for nuisance marginalisation.
            
            prior_parameter_specifications (list | dict | ParameterSet, optional): Specifications for prior parameters.
        """
        self.prior_parameter_specifications = _handle_parameter_specification(
            prior_parameter_specifications,
            _no_required_num=True
        )
        self.hyperparameter_axes    = [param_set.axes_by_type for param_set in self.prior_parameter_specifications]


        self.mixture_parameter_specifications = ParameterSet(mixture_parameter_specifications)
        self.mixture_bounds         = self.mixture_parameter_specifications.bounds


        self.log_nuisance_marg_results = log_nuisance_marg_results
        self.num_priors     = len(self.log_nuisance_marg_results)
        self.num_events     = log_nuisance_marg_results[0].shape[0]


        if not(len(self.mixture_bounds) +1 != len(log_nuisance_marg_results)):
            warnings.warn("""Number of mixtures +1 does not match number of 
priors indicated in log_nuisance_marg_results. Assigning min=0 and max=1 for remaining mixtures.""")
            
            assert len(log_nuisance_marg_results[0])>len(self.mixture_bounds) +1

            for missing_mix_idx in range(len(log_nuisance_marg_results)-(len(self.mixture_bounds) +1)):
                self.mixture_bounds.append([0., 1.])

        
        self.num_mixes              = len(self.mixture_bounds)


        # Counter for hyperparameter axes, mostly for 'index_to_hyper_parameter_info'
        hyper_idx = 0

        # Creating components of mixture for each prior
        self.num_hyper_axes         = 0
        index_to_hyper_parameter_info = {}

        for prior_idx, prior_hyper_axes in enumerate(self.hyperparameter_axes):
            idx_for_prior = 0

            update_with_defaults(prior_hyper_axes, {'spectral_parameters': {}, 'spatial_parameters':{}})

            self.num_hyper_axes+=len(prior_hyper_axes['spectral_parameters'])
            self.num_hyper_axes+=len(prior_hyper_axes['spatial_parameters'])

            for hyp_name, hyp_axis in prior_hyper_axes['spectral_parameters'].items():
                # print('hyper_idx: ', prior_idx, 'spectral_parameters', hyp_name, hyper_idx, idx_for_prior)
                index_to_hyper_parameter_info[hyper_idx] = [prior_idx, 'spectral_parameters', hyp_name, hyp_axis, idx_for_prior]
                idx_for_prior+1
                hyper_idx+=1

            for hyp_name, hyp_axis in prior_hyper_axes['spatial_parameters'].items():
                # print('hyper_idx: ', prior_idx, 'spatial_parameters', hyp_name, hyper_idx, idx_for_prior)
                index_to_hyper_parameter_info[hyper_idx] = [prior_idx, 'spatial_parameters', hyp_name, hyp_axis, idx_for_prior]
                idx_for_prior+1
                hyper_idx+=1

        self.index_to_hyper_parameter_info = index_to_hyper_parameter_info

        self.ndim = self.num_mixes + len(self.index_to_hyper_parameter_info)

        self.results = _sampler_results




    def prior_transform(self, u):
        """
        Transforms uniform samples `u` from the unit cube to the parameter space defined by mixture and hyperparameter axes.

        Args:
            u (np.array): An array of uniform samples to be transformed.

        Returns:
            np.array: Transformed samples in the parameter space.
        """

        for _mix_idx in range(self.num_mixes):
            u[_mix_idx] = self.mixture_bounds[_mix_idx][0]+u[_mix_idx]*(self.mixture_bounds[_mix_idx][1]- self.mixture_bounds[_mix_idx][0])

        for _hyper_idx in range(self.num_hyper_axes):
            hyper_axis = self.index_to_hyper_parameter_info[_hyper_idx][3]
            # Scale the value in u to the range of indices
            scaled_value = u[self.num_mixes + _hyper_idx] * len(hyper_axis)
            index = int(np.floor(scaled_value))
            # Ensure index is within bounds
            index = max(0, min(index, len(hyper_axis) - 1))
            u[self.num_mixes + _hyper_idx] = hyper_axis[index]
        return u
    
    def ln_likelihood(self, inputs):
        """
        Calculates the logarithm of the likelihood for a given set of input parameters.

        Args:
            inputs (np.array): Array of input parameters, including mixture weights and hyperparameter values.

        Returns:
            float: The logarithm of the likelihood for the given inputs.
        """
        
        mixture_weights = inputs[:self.num_mixes]
        hyper_values    = inputs[self.num_mixes:]

        slices_for_priors = [[] for idx in range(self.num_priors)]

        for hyper_idx in range(self.num_hyper_axes):
                slice_idx = np.where(self.index_to_hyper_parameter_info[hyper_idx][3]==hyper_values[hyper_idx])[0][0]
                slices_for_priors[self.index_to_hyper_parameter_info[hyper_idx][0]].append(slice_idx)

        ln_like = -np.inf
        for prior_idx in range(self.num_priors):
            ln_component = np.log(apply_direchlet_stick_breaking_direct(mixture_weights, depth=prior_idx))

            ln_marg_results_for_prior = self.log_nuisance_marg_results[prior_idx]
            try:
                ln_component += ln_marg_results_for_prior[:, *slices_for_priors[prior_idx]]
            except:
                ln_component += ln_marg_results_for_prior

            ln_like = np.logaddexp(ln_like, ln_component)

        return np.sum(ln_like)


    # Note: When allocating live points it's lower than the norm due to the discrete nature of the prior parameters
    def initiate_exploration(self, **kwargs):

        """
        Initiates the exploration process by setting up the Dynesty sampler with the class's likelihood and prior transform functions.

        Args:
            **kwargs: Keyword arguments to be passed to the Dynesty sampler.

        Returns:
            dynesty.NestedSampler: The configured Dynesty sampler object.
        """

        self.sampler = dynesty.NestedSampler(loglikelihood=self.ln_likelihood, 
                                               prior_transform=self.prior_transform, 
                                               ndim=self.ndim, 
                                               **kwargs)
        
        return self.sampler
    
    def run_exploration(self,*args, **kwargs):
        """
        Runs the nested sampling exploration process using the initialized Dynesty sampler.

        Args:
            *args: Positional arguments to be passed to the `run_nested` method of Dynesty sampler.
            **kwargs: Keyword arguments to be passed to the `run_nested` method of Dynesty sampler.

        Returns:
            dict: A dictionary containing the results of the nested sampling run.
        """
        self.run_nested_output = self.sampler.run_nested(*args, **kwargs)
        return self.run_nested_output
    

    def _pack_data(self, h5f=None, file_name=None, save_log_marg_results=True):
        """
        Private method that packs the class data into an HDF5 format.

        Equivalent to the public method for use in sub-classes.

        Args:
        h5f (h5py.File): An open HDF5 file object for writing data.
        """

        if h5f is None:
            h5f = h5py.File(file_name, 'w-')

        # Ensure mixture_parameter_specifications is properly handled if it's a ParameterSet
        if isinstance(self.mixture_parameter_specifications, ParameterSet):
            # Assuming ParameterSet has a pack method to handle its serialization
            mixture_param_group = h5f.create_group("mixture_parameter_specifications")
            self.mixture_parameter_specifications.pack(mixture_param_group)
        
        # Save log_nuisance_marg_results
        prior_parameters_group = h5f.create_group("prior_parameter_specifications")
        for prior_idx, prior_parameters in enumerate(self.prior_parameter_specifications):
            single_prior_parameters_group = prior_parameters_group.create_group(str(prior_idx))
            single_prior_parameters_group = prior_parameters.pack(single_prior_parameters_group)


        
        # Save log_nuisance_marg_results
        if save_log_marg_results:
            log_nuisance_marg_results_group = h5f.create_group("log_nuisance_marg_results")
            for result_idx, result in enumerate(self.log_nuisance_marg_results):
                log_nuisance_marg_results_group.create_dataset(str(result_idx), data=result)
        else:
            log_nuisance_marg_results_group = h5f.create_group("log_nuisance_marg_results")
            log_nuisance_marg_results_group.create_dataset(str(0), data=np.array([0.]))

        if hasattr(self.sampler, 'results'):
            sampler_group=  h5f.create_group('sampler_results')
            results = self.sampler.results
            
            # Save samples
            if hasattr(results, 'samples'):
                sampler_group.create_dataset('samples', data=np.array(results.samples))
            
            # Save log weights
            if hasattr(results, 'logwt'):
                sampler_group.create_dataset('logwt', data=np.array(results.logwt))
            
            # Save log likelihoods
            if hasattr(results, 'logl'):
                sampler_group.create_dataset('logl', data=np.array(results.logl))
            
            # Save evidence information, if available
            if hasattr(results, 'logz'):
                sampler_group.create_dataset('logz', data=np.array(results.logz))
            
            if hasattr(results, 'logzerr'):
                sampler_group.create_dataset('logzerr', data=np.array(results.logzerr))

            if hasattr(results, 'information'):
                sampler_group.create_dataset('information', data=np.array(results.information))

            if hasattr(results, 'nlive'):
                sampler_group.attrs['nlive'] = int(results.nlive)

            if hasattr(results, 'niter'):
                sampler_group.attrs['niter'] = int(results.niter)

            if hasattr(results, 'eff'):
                sampler_group.attrs['eff'] = float(results.niter)

        return h5f
    
    def pack_data(self, h5f=None, file_name=None, save_log_marg_results=False):
        return self._pack_data(h5f=h5f, file_name=file_name, save_log_marg_results=save_log_marg_results)
    

    def save(self, file_name):
        """
        Saves the class data to an HDF5 file.

        Args:
            file_name (str): The name of the file to save the data to.
        """
        h5f = self.pack_data(file_name=file_name)
        h5f.close()


    @property
    def results(self):
        if self._results == {}:
            self._results = self.sampler.results
        return self._results
    
    @results.setter
    def results(self, value):
        self._results = value


    @classmethod
    def load(cls, file_name=None):
        """
        Loads the class data from an HDF5 file.

        Args:
            file_name (str): The path to the HDF5 file to load.

        Returns:
            An instance of the class reconstructed from the file.
        """
        with h5py.File(file_name, 'r') as h5f:
            # Load mixture_parameter_specifications
            if "mixture_parameter_specifications" in h5f:
                mixture_param_group = h5f["mixture_parameter_specifications"]
                # Assuming ParameterSet class has a corresponding load or similar method
                mixture_parameter_specifications = ParameterSet.load(mixture_param_group)
            else:
                mixture_parameter_specifications = None
            
            # Load prior_parameter_specifications
            prior_parameter_specifications = []
            if "prior_parameter_specifications" in h5f:
                prior_parameters_group = h5f["prior_parameter_specifications"]
                for prior_idx in sorted(prior_parameters_group, key=int):
                    prior_group = prior_parameters_group[str(prior_idx)]
                    # Assuming ParameterSet.load is capable of handling the loading process
                    prior_parameters = ParameterSet.load(prior_group)
                    prior_parameter_specifications.append(prior_parameters)
            
            # Load log_nuisance_marg_results
            log_nuisance_marg_results = []
            log_nuisance_marg_results_group = h5f["log_nuisance_marg_results"]
                
            # Load each dataset within the "log_marg_results" group
            # Assuming the datasets are named as "0", "1", "2", ...
            # and need to be loaded in the order they were saved
            result_indices = sorted(log_nuisance_marg_results_group.keys(), key=int)
            for result_idx in result_indices:
                result = np.asarray(log_nuisance_marg_results_group[result_idx])
                log_nuisance_marg_results.append(result)

            log_nuisance_marg_results = np.asarray(log_nuisance_marg_results, dtype=object)


            _sampler_results = {}
            if 'sampler_results' in h5f:
                sampler_group = h5f['sampler_results']  # Access the group where the sampler data is stored
                
                # Load samples
                if 'samples' in sampler_group:
                    _sampler_results['samples'] = np.array(sampler_group['samples'])
                
                # Load log weights
                if 'logwt' in sampler_group:
                    _sampler_results['logwt'] = np.array(sampler_group['logwt'])
                
                # Load log likelihoods
                if 'logl' in sampler_group:
                    _sampler_results['logl'] = np.array(sampler_group['logl'])
                
                # Load evidence information, if available
                if 'logz' in sampler_group:
                    _sampler_results['logz'] = np.array(sampler_group['logz'])
                if 'logzerr' in sampler_group:
                    _sampler_results['logzerr'] = np.array(sampler_group['logzerr'])

                if 'information' in sampler_group:
                    _sampler_results['information'] = np.array(sampler_group['information'])

                if 'nlive' in sampler_group:
                    _sampler_results['nlive'] = int(sampler_group['nlive'])

                if 'niter' in sampler_group:
                    _sampler_results['niter'] = int(sampler_group['niter'])

                if 'eff' in sampler_group:
                    _sampler_results['eff'] = float(sampler_group['eff'])

            _sampler_results = ResultsWrapper(_sampler_results)
            
            
            # Reconstruct the class instance
            instance = cls(
                log_nuisance_marg_results=log_nuisance_marg_results,
                mixture_parameter_specifications=mixture_parameter_specifications,
                prior_parameter_specifications=prior_parameter_specifications,  # Assuming your __init__ can handle this
                _sampler_results=_sampler_results,
            )
            
        return instance

    
        