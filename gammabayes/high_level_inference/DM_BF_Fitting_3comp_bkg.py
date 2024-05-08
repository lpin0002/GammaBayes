import numpy as np, time, random, os, warnings, pickle
from matplotlib import pyplot as plt
from tqdm import tqdm

from gammabayes import EventData, Parameter, ParameterSet, ParameterSetCollection
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

from gammabayes.dark_matter import CombineDMComps, CustomDMRatiosModel

from gammabayes.utils import apply_direchlet_stick_breaking_direct, update_with_defaults

from gammabayes.priors import DiscreteLogPrior, log_bkg_CCR_dist, TwoCompPrior
from gammabayes.priors.astro_sources import FermiGaggeroDiffusePrior, HESSCatalogueSources_Prior

from multiprocessing import Pool
from dynesty.pool import Pool as DyPool
from dynesty import NestedSampler



class DM_BF_Fitting(object):

    def __init__(self, config_dict: dict = {}, config_file_path = None):
        print(time.strftime("DM_BF_Fitting Class initiated: %H:%M:%S"))

        try:
            self.config_dict    = config_dict
            assert (len(self.config_dict)>0) and (type(self.config_dict)==dict)
        except:
            try:
                print(f"config_file_path: {config_file_path}")
                self.config_dict = read_config_file(config_file_path)
            except:
                os.system(f"cp {os.path.dirname(__file__)+'/BF_Fitting_config_default.yaml'} {os.getcwd()}")
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
                f"SIGFRAC{self.config_dict['signal_fraction']}_CCRFRAC{self.config_dict['ccr_of_bkg_fraction']}_DIFFUSEFRAC{self.config_dict['diffuse_of_astro_fraction']}_DM_BF_Fitting_%m|%d_%H:%M")
            
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


        try:
            self.DM_Channels = self.config_dict['DM_Channels']
        except:
            self.DM_Channels = ['W+W-','ZZ','HH','tt']


        print('Channels being investigated: ', self.DM_Channels)

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

        self.NumEvents = int(round(self.config_dict['Nevents']))

        self.dm_fracs = self.config_dict['DM_Annihilation_Stick_Breaking_Ratios']
        self.bkg_fracs = self.config_dict['true_mixture_fractions']

        self.true_mixture_fractions = list(self.dm_fracs.values()) +list(self.bkg_fracs.values())

        self.nccr = int(
            round( apply_direchlet_stick_breaking_direct(self.true_mixture_fractions, depth=len(self.dm_fracs))*self.NumEvents)
                  )
                
        self.ndiffuse = int(
            round( apply_direchlet_stick_breaking_direct(self.true_mixture_fractions, depth=len(self.dm_fracs)+1)*self.NumEvents)
                  )
        
        self.npoint = int(
            round( apply_direchlet_stick_breaking_direct(self.true_mixture_fractions, depth=len(self.dm_fracs)+2)*self.NumEvents)
                  )
        

        print('BKG Nums: ', self.nccr, self.ndiffuse, self.npoint)


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
                  axes_names=['energy', 'lon', 'lat'],)
        
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

        self.DM_Models = CustomDMRatiosModel(
            channels=self.DM_Channels,
            irf_loglike=self.irf_loglike, 
            spatial_class=dark_matter_density_dist_class,
            axes=[self.energy_true_axis,
                  self.longitudeaxistrue, 
                  self.latitudeaxistrue], 
            default_spectral_parameters={'mass':config_dict['mass']})

        print("Initiating Scan Nuisance Marginalisation Hyperparameter Analysis class...")

        if 'marginalisation_bounds' in self.config_dict:
            marginalisation_bounds = self.config_dict['marginalisation_bounds']
        else:
            marginalisation_bounds = None



        self.prior_parameter_sets = list(self.DM_Models.generate_parameter_specifications(config_dict['shared_parameter_specifications']['single channel specifications']).values())
        self.mixture_parameter_set = ParameterSet(config_dict['mixture_fraction_specifications'])



        self.mass_param = config_dict['shared_parameter_specifications']['single channel specifications']['spectral_parameters']['mass']
        self.mass_param['name'] = 'mass'
        try:
            del(self.mass_param['set_name'])
        except:
            pass



        self.shared_parameters = {'mass': 
                                  [self.DM_Channels, self.mass_param]}
        # shared_parameters = {}


        self.parameter_set_collection_instance = ParameterSetCollection(
            parameter_sets = self.prior_parameter_sets,
            mixture_parameter_set = self.mixture_parameter_set,
            shared_parameters = self.shared_parameters

        )

        self.discrete_hyper_like_instance = self.discrete_scan_hyperparameter_likelihood(
            log_priors          = [*self.DM_Models.values(), self.ccr_bkg_prior, self.diffuse_astro_bkg_prior, self.hess_source_bkg_prior],
            
            log_likelihood      = self.irf_loglike, 
            log_likelihoodnormalisation = self.irf_norm_matrix,
            
            nuisance_axes       = self.ccr_bkg_prior.axes, 
            axes                = self.irf_loglike.axes,

            prior_parameter_specifications  = self.prior_parameter_sets,

            bounds = marginalisation_bounds,

            mixture_fraction_exploration_type=  self.config_dict['mixture_fraction_exploration_type'],
        )


        print("Finished Setup.")






    def simulate(self):
        print("Simulating true values...")


        self.dm_channel_true_event_data = self.DM_Models.sample_from_weights(
            num_events=self.NumEvents,
            input_weights=self.dm_fracs,)
        
        self.true_ccr_bkg_event_data  = self.ccr_bkg_prior.sample(self.nccr)

        self.true_diffuse_bkg_event_data  = self.diffuse_astro_bkg_prior.sample(self.ndiffuse)

        self.true_hess_source_bkg_event_data  = self.hess_source_bkg_prior.sample(self.npoint)

        if self.diagnostics:

            for event_data in self.dm_channel_true_event_data.values():
                event_data.peek()
                plt.show(block=self.blockplot)

            self.true_ccr_bkg_event_data.peek()
            plt.show(block=self.blockplot)

            self.true_diffuse_bkg_event_data.peek()
            plt.show(block=self.blockplot)

            self.true_hess_source_bkg_event_data.peek()
            plt.show(block=self.blockplot)


            plt.show(block=self.blockplot)

        self.bkg_event_data = self.true_ccr_bkg_event_data +self.true_diffuse_bkg_event_data +self.true_hess_source_bkg_event_data

        self.DM_true_event_data = list(self.dm_channel_true_event_data.values())[0]
        for events in list(self.dm_channel_true_event_data.values())[1:]:
            self.DM_true_event_data=self.DM_true_event_data+events


        self.DM_true_event_data.peek()
        plt.show(block=False)


        self.true_event_data = self.DM_true_event_data + self.bkg_event_data
        print("Number of true events simulated: ", len(self.true_event_data))


        print("\n\nSimulating reconstructed values...")
        if ('num_simulation_batches' in self.config_dict) and ('numcores' in self.config_dict):
            true_event_data_batches = self.true_event_data.create_batches(self.config_dict['num_simulation_batches'])
            list_of_event_data = []
            with Pool(self.config_dict['numcores']) as pool:
                for result in pool.imap(self.irf_loglike.sample,true_event_data_batches ):
                    list_of_event_data.append(result)

            self.recon_event_data = sum(list_of_event_data)

        else:
            self.recon_event_data = self.irf_loglike.sample(tqdm(self.true_event_data, total=self.true_event_data.Nevents))
        print("Finished simulating reconstructed values.\n\n")





    def nuisance_marg(self):

        keywords_in_config = ('num_marginalisation_batches' in self.config_dict) and ('numcores' in self.config_dict)
        parallelise = False
        if keywords_in_config:
            if self.config_dict['numcores']>1:
                parallelise = True 

        if parallelise:
            print("Nuisance marginalisation is being parallelised")

            batched_recon_event_data = self.recon_event_data.create_batches(self.config_dict['num_marginalisation_batches'])

            num_priors = len(self.true_mixture_fractions)+1
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
            shared_parameters                   = self.shared_parameters,

             *args, **kwargs
        )

        mass_param = self.prior_parameter_sets[2].dict_of_parameters_by_name['mass']

        fake_unitcube = np.linspace(0,1,121)

        plt.figure()
        plt.plot(fake_unitcube, mass_param.transform(fake_unitcube))
        plt.show(block=False)
        
        scan_type_sample = self.config_dict['mixture_fraction_exploration_type']=='sample'
        multiprocessing_enabled = 'numcores' in self.config_dict
        if scan_type_sample and multiprocessing_enabled:
            hyper_analysis_instance = self.discrete_hyper_like_instance.hyper_analysis_instance
            
            hyper_loglike = hyper_analysis_instance.ln_likelihood
            hyper_prior_transform = hyper_analysis_instance.prior_transform
            ndim = hyper_analysis_instance.ndim

            with DyPool(self.config_dict['numcores'], loglike=hyper_loglike, prior_transform=hyper_prior_transform) as pool:
                self.hyper_sampler = NestedSampler(
                    pool.loglike, pool.prior_transform,
                    ndim=ndim, pool=pool)
                self.hyper_sampler.run_nested(*args, **kwargs)

                pool.close() # Redundant but sometimes they leak. Currently unclear why


        else:
            self.discrete_hyper_like_instance.init_posterior_exploration()
            self.discrete_hyper_like_instance.run_posterior_exploration(*args, **kwargs)



    def run(self, path_to_measured_event_data=None):

        # if self.diagnostics:
        #     self.test_priors()

        _t1 = time.perf_counter()

        self.simulate()

        _t2 = time.perf_counter()

        self.nuisance_marg()

        _t3 = time.perf_counter()

        # self.mixture_fraction_exploration()

        # _t4 = time.perf_counter()

        print(f"Time to simulate events: {_t2-_t1:.3f} seconds")
        print(f"Time to marginalise over nuisance parameters: {_t3-_t2:.3f} seconds")
        # print(f"Time to generate hyper param log-likelihood: {_t4-_t3:.3f} seconds")



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
    





        


