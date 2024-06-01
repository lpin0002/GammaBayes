import numpy as np, time, random, warnings


from gammabayes.dark_matter.channel_spectra import (
single_channel_spectral_data_path,
PPPCReader, 
)
from gammabayes.dark_matter.density_profiles import Einasto_Profile


# $$
from gammabayes.likelihoods.irfs import IRF_LogLikelihood
from gammabayes.utils.config_utils import (
read_config_file, 
create_true_axes_from_config, 
create_recon_axes_from_config, 
)
from gammabayes.utils import (
dynamic_import,
)
from gammabayes.hyper_inference import MTree

from gammabayes import (
Parameter, ParameterSet, ParameterSetCollection,
apply_dirichlet_stick_breaking_direct,
update_with_defaults
)
from matplotlib import pyplot as plt
from gammabayes.dark_matter import CustomDMRatiosModel, CombineDMComps


from dynesty.pool import Pool as DyPool
from .setup import hl_setup_from_config



class high_level_mixture(hl_setup_from_config):

    def __init__(self, config):

        super().__init__(config=config)


    def simulate(self, Nevents=None, tree_node_values=None):

        sample_mixture_tree = self.mixture_tree.copy()
        true_event_data = []

        if not(tree_node_values is None):
            sample_mixture_tree.overwrite(tree_node_values)

        
        if Nevents is None:
            Nevents = self.Nevents_per_job


        leaf_values = list(sample_mixture_tree.leaf_values.values())

        num_samples = np.asarray(leaf_values)*Nevents

        true_event_data = [obs_prior.sample(num_sample) for obs_prior, num_sample in zip(self.observational_prior_models, num_samples)]

        # true_events = true_event_data[0]
        true_events = sum(true_event_data)




        measured_event_data = self.irf_loglike.sample(true_events)

        return true_event_data, measured_event_data
    

    def nuisance_marginalisation(self, measured_event_data):

        self.discrete_scan_hyperparameter_likelihood = dynamic_import(
            'gammabayes.hyper_inference',  
            self.config_dict['hyper_parameter_scan_class'])



        discrete_scan_hyperparameter_likelihood_instance = self.discrete_scan_hyperparameter_likelihood(
            log_priors=self.observational_prior_models,
            log_likelihood=self.irf_loglike, 
            nuisance_axes = self.true_axes, 
            axes=self.recon_axes,
            prior_parameter_specifications=self.obs_prior_parameter_specifications, 
            log_likelihoodnormalisation=self.log_irf_norm_matrix,
            bounds=[['log10', 0.5], ['linear', 0.5], ['linear', 0.5]],
            mixture_fraction_exploration_type='sample')


        reshaped_log_marg_results = discrete_scan_hyperparameter_likelihood_instance.nuisance_log_marginalisation(
            measured_event_data)
        
        return reshaped_log_marg_results

    

