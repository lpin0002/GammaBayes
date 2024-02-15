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



class ScanMarg_ConfigAnalysis(object):

    def __init__(self, config_dict: dict = {}, config_file_path = None):
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
            print(f"""An error occurred when trying to extract the seed: {excpt}\n
            A seed will be generated based on the name of the job.""")
            self.seed = int(generate_unique_int_from_string(self.jobname))
        
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



        self.nsig = int(
            round(   self.config_dict['signal_fraction']  \
                  *    self.config_dict['Nevents'])
                  )
        
        self.nccr = int(
            round((1-self.config_dict['signal_fraction']) \
                  *    self.config_dict['ccr_of_bkg_fraction']  \
                  *    self.config_dict['Nevents'])
                  )
        
        self.ndiffuse = int(
            round((1-self.config_dict['signal_fraction']) \
                  * (1-self.config_dict['ccr_of_bkg_fraction']) \
                  *    self.config_dict['diffuse_of_astro_fraction'] \
                  * self.config_dict['Nevents'])
                  )
        
        self.npoint = int(
            round((1-self.config_dict['signal_fraction']) \
                  * (1-self.config_dict['ccr_of_bkg_fraction']) \
                  * (1-self.config_dict['diffuse_of_astro_fraction']) \
                  * self.config_dict['Nevents'])
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
            log_irf_norm_matrix_path = self.config_dict['irf_norm_matrix_path']

            try:
                self.irf_norm_matrix = np.load(log_irf_norm_matrix_path)

            except Exception as excpt:
                print(f"An error occured when trying to load log irf normalisation matrices: {excpt}")
                print("Generating them based on configuration file.")
                self.log_psf_normalisations, self.log_edisp_normalisations = self.irf_loglike.create_log_norm_matrices()
                self.irf_norm_matrix = self.log_psf_normalisations + self.log_edisp_normalisations

        if 'save_irf_matrix' in self.config_dict:
            np.save(self.save_path+'irf_norm_matrx.npy', self.irf_norm_matrix)

        self.prior_parameter_sets = [ParameterSet(prior_param_specifications) for prior_param_specifications in config_dict['prior_parameter_specifications'].items()]
        self.mixture_parameter_set = ParameterSet(config_dict['mixture_fraction_specifications'])

        self.discrete_scan_hyperparameter_likelihood = dynamic_import(
            'gammabayes.hyper_inference',  
            self.config_dict['hyper_parameter_scan_class'])


        print("Constructing background priors...")

        
        self.ccr_bkg_prior = DiscreteLogPrior(
            logfunction=log_bkg_CCR_dist, 
            name='CCR Mis-identification Background',
            axes=(self.energy_true_axis, 
                  self.longitudeaxistrue, 
                  self.latitudeaxistrue,),
                  axes_names=['energy', 'lon', 'lat'], )
        
        self.diffuse_astro_bkg_prior = FermiGaggeroDiffusePrior(
            energy_axis=self.energy_true_axis, 
            longitudeaxis=self.longitudeaxistrue, 
            latitudeaxis=self.latitudeaxistrue, 
            irf=self.irf_loglike)
        
        self.hess_source_bkg_prior = HESSCatalogueSources_Prior(
            energy_axis=self.energy_true_axis, 
            longitudeaxis=self.longitudeaxistrue, 
            latitudeaxis=self.latitudeaxistrue, 
            irf=self.irf_loglike)


        print("Constructing dark matter prior...")

        
        dark_matter_density_dist_class          = dynamic_import(
            'gammabayes.dark_matter.density_profiles', 
            self.config_dict['dark_matter_density_profile'])
        
        dark_matter_spectral_class              = dynamic_import(
            'gammabayes.dark_matter.spectral_models',
              self.config_dict['dark_matter_spectral_model'])
        
        self.DM_prior = TwoCompPrior(name='Z2 Scalar Singlet dark matter',
                                spectral_class = dark_matter_spectral_class, 
                                spectral_class_kwds={'ratios':True},
                                spatial_class = dark_matter_density_dist_class,
                                irf_loglike=self.irf_loglike, 
                                axes=(self.energy_true_axis, 
                                    self.longitudeaxistrue, 
                                    self.latitudeaxistrue,), 
                                axes_names=['energy', 'lon', 'lat'],
                                default_spectral_parameters={
                                    'mass':config_dict['dark_matter_mass'],
                                    'lahS':0.1}, )        

        print("Initiating Scan Nuisance Marginalisation Hyperparameter Analysis class...")

        if 'marginalisation_bounds' in self.config_dict:
            marginalisation_bounds = self.config_dict['marginalisation_bounds']
        else:
            marginalisation_bounds = None

        self.discrete_hyper_like_instance = self.discrete_scan_hyperparameter_likelihood(
            log_priors          = (self.DM_prior, self.ccr_bkg_prior, self.diffuse_astro_bkg_prior, self.hess_source_bkg_prior),
            
            log_likelihood      = self.irf_loglike, 
            log_likelihoodnormalisation = self.irf_norm_matrix,
            
            nuisance_axes       = self.DM_prior.axes, 
            axes                = self.irf_loglike.axes,

            prior_parameter_specifications  = self.prior_parameter_sets,

            bounds = marginalisation_bounds,

            mixture_fraction_exploration_type=  self.config_dict['mixture_fraction_exploration_type'],
        )


        print("Finished Setup.")




    def test_priors(self):
        sigpriorvals = np.squeeze(self.DM_prior.construct_prior_array())
        ccrbkgpriorvals = np.squeeze(self.ccr_bkg_prior.construct_prior_array())
        diffuse_astro_bkg_prior_vals = np.squeeze(self.diffuse_astro_bkg_prior.construct_prior_array())
        hess_source_bkg_prior_vals = np.squeeze(self.hess_source_bkg_prior.construct_prior_array())

        plt.figure()
        plt.subplot(421)
        plt.title('Signal')
        plt.pcolormesh(self.longitudeaxistrue, 
                       self.latitudeaxistrue, 
                       logspace_riemann(sigpriorvals, x=self.energy_true_axis, axis=0).T )
        
        plt.subplot(422)
        plt.plot(self.energy_true_axis, 
                       iterate_logspace_integration(sigpriorvals, 
                                                    axes=[self.longitudeaxistrue, 
                                                          self.latitudeaxistrue], axisindices=(1,2)).T )
        plt.axvline(self.config_dict['dark_matter_mass'])
        plt.xscale('log')
        
        plt.subplot(423)
        plt.title('CCR BKG')

        plt.pcolormesh(self.longitudeaxistrue, 
                       self.latitudeaxistrue, 
                       logspace_riemann(ccrbkgpriorvals, x=self.energy_true_axis, axis=0).T )
        
        plt.subplot(424)
        plt.plot(self.energy_true_axis, 
                       iterate_logspace_integration(ccrbkgpriorvals, 
                                                    axes=[self.longitudeaxistrue, 
                                                          self.latitudeaxistrue], axisindices=(1,2)).T )
        plt.xscale('log')
        plt.subplot(425)
        plt.title('Diffuse Astro BKG')
        plt.pcolormesh(self.longitudeaxistrue, 
                       self.latitudeaxistrue, 
                       logspace_riemann(diffuse_astro_bkg_prior_vals, x=self.energy_true_axis, axis=0).T )
        
        plt.subplot(426)
        plt.plot(self.energy_true_axis, 
                       iterate_logspace_integration(diffuse_astro_bkg_prior_vals, 
                                                    axes=[self.longitudeaxistrue, 
                                                          self.latitudeaxistrue], axisindices=(1,2)).T )
        plt.xscale('log')        

        plt.subplot(427)
        plt.title('Point Astro BKG')
        plt.pcolormesh(self.longitudeaxistrue, 
                       self.latitudeaxistrue, 
                       logspace_riemann(hess_source_bkg_prior_vals, x=self.energy_true_axis, axis=0).T )
        
        plt.subplot(428)
        plt.plot(self.energy_true_axis, 
                       iterate_logspace_integration(hess_source_bkg_prior_vals, 
                                                    axes=[self.longitudeaxistrue, 
                                                          self.latitudeaxistrue], axisindices=(1,2)).T )
        plt.xscale('log')   

        plt.show(block=self.blockplot)


    def simulate(self):
        print("Simulating true values...")

        self.true_sig_event_data  = self.DM_prior.sample(self.nsig)
        
        self.true_ccr_bkg_event_data  = self.ccr_bkg_prior.sample(self.nccr)

        self.true_diffuse_bkg_event_data  = self.diffuse_astro_bkg_prior.sample(self.ndiffuse)

        self.true_hess_source_bkg_event_data  = self.hess_source_bkg_prior.sample(self.npoint)

        if self.diagnostics:

            self.true_sig_event_data.peek()
            plt.show(block=self.blockplot)

            self.true_ccr_bkg_event_data.peek()
            plt.show(block=self.blockplot)

            self.true_diffuse_bkg_event_data.peek()
            plt.show(block=self.blockplot)

            self.true_hess_source_bkg_event_data.peek()
            plt.show(block=self.blockplot)


            plt.show(block=self.blockplot)

        self.true_event_data = self.true_sig_event_data \
                                + self.true_ccr_bkg_event_data \
                                + self.true_diffuse_bkg_event_data \
                                + self.true_hess_source_bkg_event_data

        print("Simulating reconstructed values...")
        if ('num_simulation_batches' in self.config_dict) and ('numcores' in self.config_dict):
            true_event_data_batches = self.true_event_data.create_batches(self.config_dict['num_simulation_batches'])
            list_of_event_data = []
            with Pool(self.config_dict['numcores']) as pool:
                for result in pool.imap(self.irf_loglike.sample,true_event_data_batches ):
                    list_of_event_data.append(result)

            self.recon_event_data = sum(list_of_event_data)

        else:
            self.recon_event_data = self.irf_loglike.sample(tqdm(self.true_event_data, total=self.true_event_data.Nevents))

        if self.diagnostics:
            self.recon_event_data.peek()
            plt.show(block=self.blockplot)

            self.recon_event_data.hist_sources()
            plt.show(block=self.blockplot)



    def nuisance_marg(self):

        if ('num_marginalisation_batches' in self.config_dict) and ('numcores' in self.config_dict):
            batched_recon_event_data = self.recon_event_data.create_batches(self.config_dict['num_marginalisation_batches'])

            num_priors = 4
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



    def run(self):

        if self.diagnostics:
            self.test_priors()

        _t1 = time.perf_counter()

        self.simulate()

        _t2 = time.perf_counter()

        self.nuisance_marg()

        _t3 = time.perf_counter()

        self.mixture_fraction_exploration()

        _t4 = time.perf_counter()

        print(f"Time to simulate events: {_t2-_t1:.3f} seconds")
        print(f"Time to marginalise over nuisance parameters: {_t3-_t2:.3f} seconds")
        print(f"Time to generate hyper param log-likelihood: {_t4-_t3:.3f} seconds")



    def save(self, file_name:str, write_method='wb' ):
        """
        Saves a class instance to a pkl file.

        Args:
        file_name (str): The name of the file to save the data to.
        """

        if not(file_name.endswith('.pkl')):
            file_name = file_name+'.pkl'

        pickle.dump(self, open(file_name,write_method))

    @classmethod
    def load(cls, file_name, open_method='rb'):
        if not(file_name.endswith(".pkl")):
            file_name = file_name + ".pkl"
        return  pickle.load(open(file_name,open_method))
    

    def plot_results(self, save_fig_file_name=None, *args, **kwargs):
        from corner import corner
        from gammabayes.utils.plotting import defaults_kwargs
        from scipy.special import logsumexp
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

            bins=[41, 41, 41,*[axis.size//2 for axis in self.prior_parameter_sets[0].axes]],
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



        


