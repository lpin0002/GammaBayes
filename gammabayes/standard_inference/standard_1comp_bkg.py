import numpy as np, time, random, os, warnings, pickle
from matplotlib import pyplot as plt
from tqdm import tqdm

from gammabayes import EventData, Parameter, ParameterSet
from gammabayes.likelihoods.irfs import IRF_LogLikelihood

from gammabayes.utils.config_utils import (
    read_config_file, 
    create_true_axes_from_config, 
    create_recon_axes_from_config, 
)

from gammabayes.utils import (
    logspace_riemann, 
    iterate_logspace_integration, 
    generate_unique_int_from_string, 
    dynamic_import
)
from gammabayes.dark_matter import CombineDMComps

from gammabayes.priors import DiscreteLogPrior, log_bkg_CCR_dist, TwoCompPrior
from gammabayes.priors.astro_sources import FermiGaggeroDiffusePrior, HESSCatalogueSources_Prior

from multiprocessing import Pool
from dynesty.pool import Pool as DyPool
from dynesty import NestedSampler
from scipy.interpolate import RegularGridInterpolator
from scipy.special import logsumexp

import functools



class ScanMarg_ConfigAnalysis(object):

    def __init__(self, config_dict: dict = {}, config_file_path = None, path_to_measured_event_data=None, recon_event_data=None):
        print(time.strftime("ScanMarg_ConfigAnalysis Class initiated: %H:%M:%S"))

        try:
            self.config_dict    = config_dict
            assert (len(self.config_dict)>0) and (type(self.config_dict)==dict)
        except:
            try:
                print(f"config_file_path: {config_file_path}")
                self.config_dict = read_config_file(config_file_path)
            except:
                os.system(f"cp {os.path.dirname(__file__)+'/3COMP_BKG_config_default.yaml'} {os.getcwd()}")
                raise Exception("""No configuration dictionary or file path found. 
    A standard config file has been copied into your working directory.\n
    You can use use this path for the `config_file_path` argument of this class. Or use the `read_config_file` function 
    in Gammabayes to access it as a dictionary and pass it into the `config_dict` argument.\n""")

        try:
            self.diagnostics    = self.config_dict['diagnostics']
        except:
            self.diagnostics    = False

        try:
            self.blockplot    = self.config_dict['blockplot']
        except:
            self.blockplot    = False

        try:
            self.plot_result    = self.config_dict['plot_result']
        except:
            self.plot_result    = False

        try:
            self.jobname    = self.config_dict['jobname']
        except:
            # Yes this is a little rediculous, but at least people can find their save files
            self.jobname    = time.strftime(
                f"{self.config_dict['dark_matter_spectral_model']}_mDM{self.config_dict['dark_matter_mass']}_SIGFRAC{self.config_dict['signal_fraction']}_CCRFRAC{self.config_dict['ccr_of_bkg_fraction']}_DIFFUSEFRAC{self.config_dict['diffuse_of_astro_fraction']}_3COMP_BKG_%m|%d_%H:%M")
            
        try:
            self.seed       = int(self.config_dict['seed'])
        except Exception as excpt:

            self.seed = int(generate_unique_int_from_string(self.jobname))

            print(f"""An error occurred when trying to extract the seed: {excpt}\n
            A seed has been generated based on the name of the job with value: {self.seed}""")
        
        try:
            self.numjobs    = self.config_dict['numjobs']
        except:
            self.numjobs    = 1

        # The larger this bugger the more values are considered at once when vectorised
            # This also leads to a subsequent increase in memory. Currently this must be
            # chosen with a bit of trial and error
        try:
            self.mixture_scanning_buffer = self.config_dict['mixture_scanning_buffer']
        except:
            self.mixture_scanning_buffer = 10


        ########################################################################
        ########################################################################
        #### Setting the random seed for calculations
        random.seed(self.seed)

        try:
            self.save_path = self.config_dict['save_path']
            if not(os.path.exists(self.save_path)):
                folderstructure = self.save_path.split('/')
                folderpath = folderstructure[0]
                os.makedirs(folderpath, exist_ok=True)
                for folder in folderstructure[1:-1]:
                    folderpath = f"{folderpath}/{folder}"
                    os.makedirs(folderpath, exist_ok=True)


        except Exception as excpt:
            print(f"An error occured when trying to unpack save path: {excpt}")
            self.save_path = f'data/{self.jobname}/'
            os.makedirs('data', exist_ok=True)
            os.makedirs(f'data/{self.jobname}', exist_ok=True)



        if 'path_to_measured_event_data' in config_dict:
            self.path_to_measured_event_data = config_dict['path_to_measured_event_data']
        else:
            self.path_to_measured_event_data = path_to_measured_event_data


        if self.path_to_measured_event_data is not None:

            self.recon_event_data = EventData.load(self.path_to_measured_event_data)
        elif recon_event_data is not None:
            self.recon_event_data = recon_event_data

        else:
            self.recon_event_data = None
            self.nsig = int(
                round( self.config_dict['signal_fraction'] * self.config_dict['Nevents'])
                    )
            
            self.nbkg = int(
                round( (1-self.config_dict['signal_fraction']) * self.config_dict['Nevents'])
                    )
            
        
        self.energy_true_axis,  self.longitudeaxistrue, self.latitudeaxistrue = create_true_axes_from_config(self.config_dict)
        self.energy_recon_axis, self.longitudeaxis,     self.latitudeaxis     = create_recon_axes_from_config(self.config_dict)


        self.irf_loglike = IRF_LogLikelihood(
            axes   =   [self.energy_recon_axis,    
                        self.longitudeaxis,     
                        self.latitudeaxis], 
            dependent_axes =   [self.energy_true_axis,     
                                self.longitudeaxistrue, 
                                self.latitudeaxistrue])

        if 'common_norm_matrices' in self.config_dict:
            self.common_norm_matrices = self.config_dict['common_norm_matrices']
        else:
            print('common_norm_matrices key not found in configuration dictionary')
            self.common_norm_matrices = False


        if not(self.common_norm_matrices):
            print("Going to create normalisations...")
            self.log_psf_normalisations, self.log_edisp_normalisations = self.irf_loglike.create_log_norm_matrices()
            self.irf_norm_matrix = self.log_psf_normalisations + self.log_edisp_normalisations


        else:
            if 'irf_norm_matrix_path' in self.config_dict:
                log_irf_norm_matrix_path = self.config_dict['irf_norm_matrix_path']

                self.irf_norm_matrix = np.load(log_irf_norm_matrix_path)


            elif 'log_psf_norm_matrix_path' in self.config_dict:
                self.log_psf_norms = np.load(self.config_dict["log_psf_norm_matrix_path"])
                self.log_edisp_norms = np.load(self.config_dict["log_edisp_norm_matrix_path"])
                self.irf_norm_matrix = self.log_psf_norms + self.log_edisp_norms
            else:
                raise Exception("Neither 'irf_norm_matrix_path' or 'log_psf_norm_matrix_path' are within the configuration dictionary but shared matrices is switched on.")


        if 'save_irf_matrix' in self.config_dict:
            if 'irf_norm_matrix_path' in self.config_dict:
                np.save('irf_norm_matrix_path', self.irf_norm_matrix)
            else:
                np.save(self.save_path+'irf_norm_matrix.npy', self.irf_norm_matrix)

        self.prior_parameter_sets = [ParameterSet(prior_param_specifications) for prior_param_specifications in config_dict['prior_parameter_specifications'].items()]
        self.mixture_parameter_set = ParameterSet(config_dict['mixture_fraction_specifications'])

        self.discrete_scan_hyperparameter_likelihood = dynamic_import(
            'gammabayes.hyper_inference',  
            self.config_dict['hyper_parameter_scan_class'])


        print("Constructing background prior...")

        nuisance_axes = [self.energy_true_axis,     
                                self.longitudeaxistrue, 
                                self.latitudeaxistrue]
        nuisance_axes_meshes = np.meshgrid(*nuisance_axes, indexing='ij')
        nuisance_axes_flattened = np.array([axis.flatten() for axis in nuisance_axes_meshes])


        log_bkg_CCR_dist_log_vals = log_bkg_CCR_dist(*nuisance_axes_meshes)


        from gammabayes.priors.astro_sources import construct_fermi_gaggero_matrix, construct_hess_source_map
        construct_fermi_gaggero_matrix_log_vals = np.log(construct_fermi_gaggero_matrix(*nuisance_axes, log_aeff=self.irf_loglike.log_aeff))
        construct_hess_source_map_log_vals = np.log(construct_hess_source_map(*nuisance_axes, log_aeff=self.irf_loglike.log_aeff))


        log_bkg_dist_vals = logsumexp([log_bkg_CCR_dist_log_vals, 
                                       construct_fermi_gaggero_matrix_log_vals, 
                                       construct_hess_source_map_log_vals], axis=0)

        bkg_interpolator_tuple_inputs = RegularGridInterpolator(
            values=log_bkg_dist_vals, 
            points=nuisance_axes)


        partial_bkg_interpolator = functools.partial(self.bkg_interpolator, interpolator=bkg_interpolator_tuple_inputs)

        
        self.bkg_prior = DiscreteLogPrior(
            logfunction=partial_bkg_interpolator, 
            name='CCR Mis-identification Background',
            axes=(self.energy_true_axis, 
                  self.longitudeaxistrue, 
                  self.latitudeaxistrue,),
                  axes_names=['energy', 'lon', 'lat'], )
        
        print("Constructing dark matter prior...")

        
        dark_matter_density_dist_class          = dynamic_import(
            'gammabayes.dark_matter.density_profiles', 
            self.config_dict['dark_matter_density_profile'])
        
        dark_matter_spectral_class              = dynamic_import(
            'gammabayes.dark_matter.spectral_models',
              self.config_dict['dark_matter_spectral_model'])
        
        self.DM_prior = TwoCompPrior(name='Dark matter prior',
                                spectral_class = dark_matter_spectral_class, 
                                spectral_class_kwds={'ratios':True},
                                spatial_class = dark_matter_density_dist_class,
                                irf_loglike=self.irf_loglike, 
                                axes=(self.energy_true_axis, 
                                    self.longitudeaxistrue, 
                                    self.latitudeaxistrue,), 
                                axes_names=['energy', 'lon', 'lat'],
                                default_spectral_parameters=config_dict['signal_default_spectral_parameters'], 
                                default_spatial_parameters=config_dict['signal_default_spatial_parameters'], 
                                )
                

        print("Initiating Scan Nuisance Marginalisation Hyperparameter Analysis class...")

        if 'marginalisation_bounds' in self.config_dict:
            marginalisation_bounds = self.config_dict['marginalisation_bounds']
        else:
            marginalisation_bounds = None

        self.discrete_hyper_like_instance = self.discrete_scan_hyperparameter_likelihood(
            log_priors          = (self.DM_prior, self.bkg_prior,),
            
            log_likelihood      = self.irf_loglike, 
            log_likelihoodnormalisation = self.irf_norm_matrix,
            
            nuisance_axes       = self.DM_prior.axes, 
            axes                = self.irf_loglike.axes,

            prior_parameter_specifications  = self.prior_parameter_sets,

            bounds = marginalisation_bounds,

            mixture_fraction_exploration_type=  self.config_dict['mixture_fraction_exploration_type'],
        )


        print("Finished Setup.")


    def bkg_interpolator(self, energy, longitude, latitude, spectral_parameters={}, spatial_parameters={}, interpolator=None):
        return interpolator((energy, longitude, latitude))


    def test_priors(self):
        sigpriorvals = np.squeeze(self.DM_prior.construct_prior_array())
        bkgpriorvals = np.squeeze(self.bkg_prior.construct_prior_array())

        plt.figure()
        plt.subplot(221)
        plt.title('Signal')
        plt.pcolormesh(self.longitudeaxistrue, 
                       self.latitudeaxistrue, 
                       logspace_riemann(sigpriorvals, x=self.energy_true_axis, axis=0).T )
        
        plt.subplot(222)
        plt.plot(self.energy_true_axis, 
                       iterate_logspace_integration(sigpriorvals, 
                                                    axes=[self.longitudeaxistrue, 
                                                          self.latitudeaxistrue], axisindices=(1,2)).T )
        plt.axvline(self.config_dict['dark_matter_mass'])
        plt.xscale('log')
        
        plt.subplot(223)
        plt.title('CCR BKG')

        plt.pcolormesh(self.longitudeaxistrue, 
                       self.latitudeaxistrue, 
                       logspace_riemann(bkgpriorvals, x=self.energy_true_axis, axis=0).T )
        
        plt.subplot(224)
        plt.plot(self.energy_true_axis, 
                       iterate_logspace_integration(bkgpriorvals, 
                                                    axes=[self.longitudeaxistrue, 
                                                          self.latitudeaxistrue], axisindices=(1,2)).T )
        plt.xscale('log')

        plt.show(block=self.blockplot)


    def simulate(self, nsig=None, nbkg=None, event_save_path=None):
        print("Simulating true values...")

        if event_save_path is None:
            if 'save_path_for_measured_event_data' in self.config_dict:
                event_save_path = self.config_dict['save_path_for_measured_event_data']

        if nsig is None:
            nsig = self.nsig

        if nbkg is None:
            nbkg = self.nbkg

        true_sig_event_data  = self.DM_prior.sample(nsig)
        
        true_bkg_event_data  = self.bkg_prior.sample(nbkg)


        if self.diagnostics:

            true_sig_event_data.peek()
            plt.show(block=self.blockplot)

            true_bkg_event_data.peek()
            plt.show(block=self.blockplot)


        true_event_data = true_sig_event_data + true_bkg_event_data

        print("Simulating reconstructed values...")
        if ('num_simulation_batches' in self.config_dict) and ('numcores' in self.config_dict):
            true_event_data_batches = true_event_data.create_batches(self.config_dict['num_simulation_batches'])
            list_of_event_data = []
            with Pool(self.config_dict['numcores']) as pool:
                for result in pool.imap(self.irf_loglike.sample,true_event_data_batches ):
                    list_of_event_data.append(result)

            recon_event_data = sum(list_of_event_data)

        else:
            recon_event_data = self.irf_loglike.sample(tqdm(true_event_data, total=true_event_data.Nevents))

        if self.diagnostics:
            recon_event_data.peek()
            plt.show(block=self.blockplot)

            recon_event_data.hist_sources()
            plt.show(block=self.blockplot)

        if self.recon_event_data is None:
            self.recon_event_data = recon_event_data


        if event_save_path is not None:
            recon_event_data.save(filename=event_save_path)

        return recon_event_data



    def nuisance_marg(self):

        if ('num_marginalisation_batches' in self.config_dict) and ('numcores' in self.config_dict):
            batched_recon_event_data = self.recon_event_data.create_batches(self.config_dict['num_marginalisation_batches'])

            num_priors = len(self.discrete_hyper_like_instance.log_priors)
            list_of_log_nuisance_marg_results = []

            
            with Pool(self.config_dict['numcores']) as pool:
                for result in pool.map(self.discrete_hyper_like_instance.nuisance_log_marginalisation, 
                                       batched_recon_event_data):
                    
                    list_of_log_nuisance_marg_results.append(result)



            log_nuisance_marg_results = list_of_log_nuisance_marg_results[0]
            for batch_idx, log_marg_result in enumerate(list_of_log_nuisance_marg_results[1:]):
                for prior_idx in range(num_priors):
                    log_nuisance_marg_results[prior_idx] = np.append(
                        log_nuisance_marg_results[prior_idx], 
                        log_marg_result[prior_idx],
                        axis=0
                        )

            self.log_nuisance_marg_results = log_nuisance_marg_results

        else:
            self.log_nuisance_marg_results = self.discrete_hyper_like_instance.nuisance_log_marginalisation(
                measured_event_data=self.recon_event_data)
            
        self.discrete_hyper_like_instance.log_nuisance_marg_results = self.log_nuisance_marg_results
        
        return self.log_nuisance_marg_results
    

    def mixture_fraction_exploration(self,  *args, **kwargs):


        self.discrete_hyper_like_instance.select_scan_output_exploration_class(
            log_nuisance_marg_results           = self.log_nuisance_marg_results,
            mixture_parameter_specifications    = self.mixture_parameter_set,
            prior_parameter_specifications      = self.prior_parameter_sets,
             *args, **kwargs
        )
        
        scan_type_sample = self.config_dict['mixture_fraction_exploration_type']=='sample'
        multiprocessing_enabled = 'numcores' in self.config_dict
        if scan_type_sample and multiprocessing_enabled:
            scan_output_exploration_class_instance = self.discrete_hyper_like_instance.hyper_analysis_instance
            
            ptform = scan_output_exploration_class_instance.prior_transform
            loglike = scan_output_exploration_class_instance.ln_likelihood
            ndim = scan_output_exploration_class_instance.ndim

            with DyPool(self.config_dict['numcores'], loglike, ptform) as pool:
                self.discrete_hyper_like_instance.hyper_analysis_instance.sampler = NestedSampler(
                    pool.loglike, pool.prior_transform,
                    ndim, pool=pool)
                self.discrete_hyper_like_instance.run_posterior_exploration(*args, **kwargs)

                pool.close() # Redundant but sometimes they leak. Currently unclear why


        else:
            self.discrete_hyper_like_instance.init_posterior_exploration()
            self.discrete_hyper_like_instance.run_posterior_exploration(*args, **kwargs)



    def run(self, path_to_measured_event_data=None, event_save_path=None):

        if self.diagnostics:
            self.test_priors()

        _t1 = time.perf_counter()


        if (self.recon_event_data is None):
            if (path_to_measured_event_data is None):
                self.simulate(event_save_path=event_save_path)
            else:
                self.recon_event_data = EventData.load(path_to_measured_event_data)
                self.path_to_measured_event_data = path_to_measured_event_data
        
        _t2 = time.perf_counter()

        self.nuisance_marg()

        _t3 = time.perf_counter()

        # self.mixture_fraction_exploration()

        # _t4 = time.perf_counter()

        print(f"Time to simulate events: {_t2-_t1:.3f} seconds")
        print(f"Time to marginalise over nuisance parameters: {_t3-_t2:.3f} seconds")
        # print(f"Time to generate hyper param log-likelihood: {_t4-_t3:.3f} seconds")



    def save(self, filename:str, write_method='wb' ):
        """
        Saves a class instance to a pkl file.

        Args:
        file_name (str): The name of the file to save the data to.
        """

        if not(filename.endswith('.pkl')):
            filename = filename+'.pkl'

        pickle.dump(self, open(filename,write_method))

    @classmethod
    def load(cls, filename, open_method='rb'):
        if not(filename.endswith(".pkl")):
            filename = filename + ".pkl"
        return  pickle.load(open(filename,open_method))
    

    def plot_results(self, save_fig_file_name=None, *args, **kwargs):
        from corner import corner
        from gammabayes.utils.plotting import defaults_kwargs
        defaults_kwargs['smooth'] = 2



        num_params = 5
        sampling_results = self.discrete_hyper_like_instance.posterior_exploration_results

        fig = plt.figure(figsize=(12,12))
        figure=corner(sampling_results.samples_equal(), fig=fig,
            labels=['sig/total', 'ccr/bkg', 'diffuse/astro bkg', r'$m_{\chi}$ [TeV]', r'$\alpha$'],
            truths=[self.config_dict['signal_fraction'],
                    self.config_dict['ccr_of_bkg_fraction'],
                    self.config_dict['diffuse_of_astro_fraction'],
                    self.config_dict['dark_matter_mass'],
                    0.17,],

            quantiles=[0.025, .16, 0.5, .84, 0.975],

            bins=[41, 41, 41,*[axis.size for axis in self.prior_parameter_sets[0].axes]],
            #    range=([0.44,0.68], [0.7,0.9], [0.3,0.5], *[[axis.min(), axis.max()] for axis in prior_parameters.axes]),
            axes_scale=['linear', 'linear', 'linear', 'log', 'log'],
            
            **defaults_kwargs)

        for ax_idx, ax in enumerate(figure.get_axes()):
            # Find lines representing the quantiles (the 3rd line for each histogram is the median, based on the ordering in `quantiles`)
            lines = ax.get_lines()
            if (len(lines) > 2):
                if (ax_idx%(num_params+1)==0): 
                    for line_idx, line in enumerate(lines): # Check if there are enough lines (for histograms)
                        if line_idx==2:
                            line.set_color('tab:green')  # Change the color of the median lines
                        elif line_idx<len(lines)-1:
                            line.set_color('tab:blue')
        plt.suptitle(str(self.recon_event_data.Nevents) + " events", size=24)

        plt.tight_layout()
        if save_fig_file_name is not None:
            plt.savefig(save_fig_file_name)

        plt.show()



        


