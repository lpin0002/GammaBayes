from gammabayes import GammaObs, GammaObsCube, GammaLogExposure, GammaBinning
from gammabayes.priors import DiscreteLogPrior, SourceFluxDiscreteLogPrior
from gammabayes.likelihoods.irfs import IRF_LogLikelihood
from gammabayes.hyper_inference import MTree
from gammabayes.utils import iterate_logspace_integration
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from icecream import ic
class SimulationContainer:


    def __init__(self, priors:list[DiscreteLogPrior] | list[SourceFluxDiscreteLogPrior], 
                 irf_loglikes, 
                 mixture_tree,
                 true_binning_geometry,
                 recon_binning_geometry,
                 log_exposures: GammaLogExposure | list[GammaLogExposure], 
                 Nevents: int|list=1, 
                 observation_times= 0.5*u.hr, 
                 observation_meta=None,
                 pointing_dirs=None,
                 rescale_priors=True):
        self.true_binning_geometry = true_binning_geometry
        self.recon_binning_geometry = recon_binning_geometry
        self.mixture_tree = mixture_tree
        self.priors = priors
        self.log_exposures = log_exposures
        self.irf_loglikes = irf_loglikes
        self.pointing_dirs = pointing_dirs
        self.observation_times = observation_times
        self.Nevents = Nevents


        combined_log_exposure = self.setup_log_exposures()

        if rescale_priors:
            self.rescale_source_flux_priors(mixture_tree=self.mixture_tree, 
                                            Nevents=self.Nevents, 
                                            true_binning_geometry=true_binning_geometry,
                                            combined_log_exposure=combined_log_exposure)




        self._setup_observation_containers()
        self._setup_observation_priors()


    def setup_log_exposures(self):

        if isinstance(self.log_exposures, GammaLogExposure):
            self.log_exposures = [self.log_exposures]


        self.combined_log_exposure = sum(self.log_exposures)

        if isinstance(self.Nevents, (int, float)):
            self.Nevents = [int(round(self.Nevents/len(self.log_exposures)))]*len(self.log_exposures)

        elif isinstance(self.Nevents, (np.ndarray, list)) and (len(self.Nevents)!=len(self.log_exposures)):
            raise ValueError("Length of Nevents is not single value or same length as log_exposures.")
        

        return self.combined_log_exposure





    def calc_Nevents(self, true_binning_geometry: GammaBinning = None, combined_log_exposure:GammaLogExposure=None):

        if true_binning_geometry is None:
            true_binning_geometry = self.true_binning_geometry

        if combined_log_exposure is None:
            combined_log_exposure = self.combined_log_exposure

        event_counts = {}

        for prior in self.priors:
            if isinstance(prior, SourceFluxDiscreteLogPrior):
                log_flux_matrix = prior.log_source_flux_mesh_efficient(energy=true_binning_geometry.energy_axis, lon=true_binning_geometry.lon_axis, lat=true_binning_geometry.lat_axis)
            elif isinstance(prior, DiscreteLogPrior):
                if hasattr(prior, "log_mesh_efficient_func"):
                    log_flux_matrix = prior.log_mesh_efficient_func(energy=true_binning_geometry.energy_axis, lon=true_binning_geometry.lon_axis, lat=true_binning_geometry.lat_axis)
                else:
                    energy_axis_mesh, longitude_axis_mesh, latitude_axis_mesh = true_binning_geometry.axes_mesh
                    log_flux_matrix = prior(energy=energy_axis_mesh, lon=longitude_axis_mesh, lat=latitude_axis_mesh)

            log_flux_matrix=combined_log_exposure+log_flux_matrix

            log_event_count  = iterate_logspace_integration(logy=log_flux_matrix, axes=[*true_binning_geometry.axes,])


            event_counts[prior.name] = int(round(np.exp(log_event_count)))

        return event_counts


    def rescale_source_flux_priors(self, mixture_tree:MTree, Nevents:dict[str:int], true_binning_geometry: GammaBinning = None, combined_log_exposure:GammaLogExposure=None):
        unadjusted_event_counts = self.calc_Nevents(true_binning_geometry=true_binning_geometry, combined_log_exposure=combined_log_exposure)


        total_events = sum(self.Nevents)

        leaf_values = mixture_tree.leaf_values

        events_by_prior = {prior_name:total_events*leaf_value for prior_name, leaf_value in leaf_values.items()}

        event_scaling_factors = {prior_name:num_events/unadjusted_event_counts[prior_name] for prior_name, num_events in events_by_prior.items()}

        for prior_idx, prior in enumerate(self.priors):
            self.priors[prior_idx].rescale(np.log(event_scaling_factors[prior.name]))


        self.rescaled = True



    
        
    def _setup_observation_containers(self):

        

        if isinstance(self.observation_times, (u.Quantity, float)):
            self.observation_times = [self.observation_times]*len(self.log_exposures)
        elif isinstance(self.observation_times, (np.ndarray, list)):
            if len(self.observation_times)!=len(self.log_exposures):
                raise ValueError("Length of observation times is not single value or same length as log_exposures.")
            
            
        if isinstance(self.pointing_dirs, (u.Quantity)):
            self.pointing_dirs = [self.pointing_dirs]*len(self.log_exposures)
        elif isinstance(self.pointing_dirs, np.ndarray):
            if self.pointing_dirs.ndim ==1:
                self.pointing_dirs = [self.pointing_dirs]*len(self.log_exposures)
            elif self.pointing_dirs.ndim==2:
                if len(self.pointing_dirs)!=len(self.log_exposures):
                    raise ValueError("Length of pointing directions is not single value or same length as log_exposures.")
        elif isinstance(self.pointing_dirs, list):
            if len(self.pointing_dirs)!=len(self.log_exposures):
                raise ValueError("Length of pointing directions is not single value or same length as log_exposures.")


        if isinstance(self.irf_loglikes, IRF_LogLikelihood) or isinstance(self.irf_loglikes, float):
            self.irf_loglikes = [self.irf_loglikes]*len(self.log_exposures)
        elif isinstance(self.irf_loglikes, (np.ndarray, list)):
            if len(self.irf_loglikes)!=len(self.log_exposures):
                raise ValueError("Length of irf loglikelihoods is not single value or same length as log_exposures.")




            



        # TODO: #TechDebt
        self.observation_containers = {f"Observation_{obs_idx}":{
            "true_binning_geometry":self.true_binning_geometry, 
            "recon_binning_geometry":self.recon_binning_geometry, 
            "pointing_dir": self.pointing_dirs[obs_idx],
            "irf_loglike": self.irf_loglikes[obs_idx],
            "log_exposure": self.log_exposures[obs_idx],
            "Nevents": self.Nevents[obs_idx],
            } for obs_idx in range(len(self.log_exposures))}





    def _setup_observation_priors(self):
        for observation_name in self.observation_containers.keys():
            priors_for_obs = []
            for prior in self.priors:
                if isinstance(prior, SourceFluxDiscreteLogPrior):
                    prior.log_exposure_map = self.observation_containers[observation_name]["log_exposure"]
                    priors_for_obs.append(prior)
                else:
                    priors_for_obs.append(prior)


            self.observation_containers[observation_name]['priors'] = priors_for_obs



    def simulate_true_observation(self, 
                                  log_exposure, 
                                  priors, 
                                  mixture_tree:MTree, 
                                  Nevents:list[int]=None, tree_node_values=None, 
                                  **kwargs) -> dict[GammaObs]:



        sample_mixture_tree = mixture_tree.copy()
        true_event_data = []

        if not(tree_node_values is None):
            sample_mixture_tree.overwrite(tree_node_values)

    

        leaf_values = list(sample_mixture_tree.leaf_values.values())

        num_samples = np.asarray(leaf_values)*Nevents

        true_event_data = {}

        for prior, num_sample in zip(priors, num_samples):
            if hasattr(prior, "log_exposure"):
                prior.log_exposure = log_exposure

            true_event_data[prior.name] = prior.sample(num_sample)
            
        return true_event_data
    

    def simulate_measured_observation(self, irf_loglike:IRF_LogLikelihood, true_event_obs:GammaObs):

        return irf_loglike.sample(true_event_obs)



    def simulation_observation_set(self):
        
        for observation_name, observation_info in tqdm(self.observation_containers.items()):

            true_event_data = self.simulate_true_observation(mixture_tree=self.mixture_tree, **observation_info)
            self.observation_containers[observation_name]["true_event_data"] = true_event_data

            combined_true_event_data = sum(list(true_event_data.values()))

            measured_event_data = self.simulate_measured_observation(irf_loglike=observation_info['irf_loglike'], 
                                                                     true_event_obs=combined_true_event_data)
            
            self.observation_containers[observation_name]["measured_event_data"] = measured_event_data




    def peek(self, *args, **kwargs):


        true_event_data = []
        measured_event_data = []

        if "figsize" not in kwargs:
            kwargs["figsize"] = (12, 12)




        for observation_name, observation_info in self.observation_containers.items():

            if "true_event_data" in observation_info:
                true_event_data.append(sum(list(observation_info["true_event_data"].values())))

            if "measured_event_data" in observation_info:
                measured_event_data.append(observation_info["measured_event_data"])

        combined_true_event_data = sum(true_event_data)
        combined_measured_event_data = sum(measured_event_data)


        fig, ax = plt.subplots(nrows=2, ncols=2, *args, **kwargs)

        subplot_kwargs = kwargs

        subplot_kwargs["figsize"] = (subplot_kwargs["figsize"][0], subplot_kwargs["figsize"][1]/2)

        combined_true_event_data.peek(axs=ax[0], **kwargs)
        combined_measured_event_data.peek(axs=ax[1], **kwargs)








