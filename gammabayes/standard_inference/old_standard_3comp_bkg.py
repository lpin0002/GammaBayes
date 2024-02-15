

from gammabayes.likelihoods.irfs import irf_loglikelihood
from gammabayes.priors.astro_sources import (
    construct_hess_source_map_interpolation, 
    construct_log_fermi_gaggero_bkg,
)
from gammabayes.priors import discrete_logprior, log_bkg_CCR_dist

from gammabayes.utils.plotting import logdensity_matrix_plot
from gammabayes.utils.config_utils import (
    read_config_file, 
    create_true_axes_from_config, 
    create_recon_axes_from_config, 
    construct_parameter_axes,
    iterative_parameter_axis_construction
)
from gammabayes.utils import (
    save_to_pickle, 
    logspace_riemann, 
    iterate_logspace_integration, 
    bin_centres_to_edges, 
    generate_unique_int_from_string, 
    extract_axes,
    dynamic_import
)

from gammabayes.dark_matter import combine_DM_models
from gammabayes.dark_matter.density_profiles import Einasto_Profile


from scipy import special
from tqdm import tqdm
import os, sys, time, warnings, random, numpy as np, matplotlib.pyplot as plt



class standard_3COMP_BKG(object):
        
    def __init__(self, config_dict: dict = {}, config_file_path = None):
        try:
            self.config_dict    = config_dict
            assert (len(self.config_dict)>0) and (type(self.config_dict)==dict)
        except:
            try:
                print(f"config_file_path: {config_file_path}")
                self.config_dict = read_config_file(config_file_path)
            except:
                os.system(f"cp {os.path.dirname(__file__)+'/Z2_DM_3COMP_BKG_config_default.yaml'} {os.getcwd()}")
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
            self.jobname    = time.strftime(f"DM{self.config_dict['dark_matter_mass']}_SIGFRAC{self.config_dict['signalfraction']}_CCRFRAC{self.config_dict['ccr_of_bkg_fraction']}_DIFFUSEFRAC{self.config_dict['diffuse_of_astro_fraction']}_3COMP_BKG_%m|%d_%H:%M")
            
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


        try:
            self.source_hyperparameter_input = iterative_parameter_axis_construction(self.config_dict['parameter_scan_specifications'])
        except:
            self.source_hyperparameter_input = None


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



        self.nsig               = int(round(   self.config_dict['signalfraction']  *    self.config_dict['Nevents']))
        self.nccr               = int(round((1-self.config_dict['signalfraction']) *    self.config_dict['ccr_of_bkg_fraction']  *    self.config_dict['Nevents']))
        self.ndiffuse           = int(round((1-self.config_dict['signalfraction']) * (1-self.config_dict['ccr_of_bkg_fraction']) *    self.config_dict['diffuse_of_astro_fraction']  * self.config_dict['Nevents']))
        self.npoint             = int(round((1-self.config_dict['signalfraction']) * (1-self.config_dict['ccr_of_bkg_fraction']) * (1-self.config_dict['diffuse_of_astro_fraction']) * self.config_dict['Nevents']))


        self.energy_true_axis,  self.longitudeaxistrue, self.latitudeaxistrue       = create_true_axes_from_config(self.config_dict)
        self.energy_recon_axis, self.longitudeaxis,     self.latitudeaxis           = create_recon_axes_from_config(self.config_dict)


        self.irf_loglike = irf_loglikelihood(axes   =   [self.energy_recon_axis,    self.longitudeaxis,     self.latitudeaxis], 
                                     dependent_axes =   [self.energy_true_axis,     self.longitudeaxistrue, self.latitudeaxistrue])

        try:
            self.common_norm_matrices = self.config_dict['common_norm_matrices']
        except:
            self.common_norm_matrices = False

        if not(self.common_norm_matrices):
            print("Going to create normalisations...")
            self.log_psf_normalisations, self.log_edisp_normalisations = self.irf_loglike.create_log_norm_matrices()
            self.irf_norm_matrix = self.log_psf_normalisations + self.log_edisp_normalisations
        else:
            log_psf_norm_matrix_path = self.config_dict['log_psf_norm_matrix_path']
            log_edisp_norm_matrix_path = self.config_dict['log_edisp_norm_matrix_path']
            try:
                self.log_psf_normalisations, self.log_edisp_normalisations = np.load(log_psf_norm_matrix_path), np.load(log_edisp_norm_matrix_path)
                self.irf_norm_matrix = self.log_psf_normalisations + self.log_edisp_normalisations
            except Exception as excpt:
                print(f"An error occured when trying to load log irf normalisation matrices: {excpt}")
                print("Generating them based on configuration file.")
                self.log_psf_normalisations, self.log_edisp_normalisations = self.irf_loglike.create_log_norm_matrices()
                self.irf_norm_matrix = self.log_psf_normalisations + self.log_edisp_normalisations





        print("Constructing background priors...")

        
        self.ccr_bkg_prior = discrete_logprior(logfunction=log_bkg_CCR_dist, name='CCR Mis-identification Background Prior',
                               axes=(   self.energy_true_axis, 
                                        self.longitudeaxistrue, 
                                        self.latitudeaxistrue,), 
                                    axes_names=['energy', 'lon', 'lat'], )
        
        log_fermi_gaggero_bkg_class_instance = construct_log_fermi_gaggero_bkg(energy_axis=self.energy_true_axis, 
                                                                                   longitudeaxis=self.longitudeaxistrue, 
                                                                                   latitudeaxis=self.latitudeaxistrue, 
                                                                                   log_aeff=self.irf_loglike.log_aeff)
        
        self.diffuse_astro_bkg_prior = discrete_logprior(logfunction=log_fermi_gaggero_bkg_class_instance.log_fermi_diffuse_func, name='Diffuse Astrophysical Background Prior',
                               axes=(   self.energy_true_axis, 
                                        self.longitudeaxistrue, 
                                        self.latitudeaxistrue,), 
                                    axes_names=['energy', 'lon', 'lat'], )
        
        log_hess_bkg_class_instance = construct_hess_source_map_interpolation(energy_axis=self.energy_true_axis, 
                                                                                   longitudeaxis=self.longitudeaxistrue, 
                                                                                   latitudeaxis=self.latitudeaxistrue, 
                                                                                   log_aeff=self.irf_loglike.log_aeff)
        
        self.point_astro_bkg_prior = discrete_logprior(logfunction=log_hess_bkg_class_instance.log_astro_func, name='Point Source Astrophysical Background Prior',
                               axes=(   self.energy_true_axis, 
                                        self.longitudeaxistrue, 
                                        self.latitudeaxistrue,), 
                                    axes_names=['energy', 'lon', 'lat'], )


        print("Constructing dark matter prior...")


        dark_matter_spectral_class = dynamic_import('gammabayes.dark_matter.models', self.config_dict['dark_matter_spectral_model'])
        self.discrete_hyperparameter_likelihood = dynamic_import('gammabayes.hyper_inference', self.config_dict['hyper_parameter_scan_class'])

        SS_DM_combine_instance = combine_DM_models(dark_matter_spectral_class, 
                                                   Einasto_Profile, 
                                                   self.irf_loglike, spectral_class_kwds={'ratios':True})
        self.logDMpriorfunc, self.logDMpriorfunc_mesh_efficient = SS_DM_combine_instance.DM_signal_dist, SS_DM_combine_instance.DM_signal_dist_mesh_efficient

        self.DM_prior = discrete_logprior(logfunction=self.logDMpriorfunc, 
                             log_mesh_efficient_func=self.logDMpriorfunc_mesh_efficient, 
                             name='Scalar Singlet Dark Matter Prior',
                               axes=(   self.energy_true_axis, 
                                        self.longitudeaxistrue, 
                                        self.latitudeaxistrue,), 
                             axes_names=['energy', 'lon', 'lat'],
                             default_spectral_parameters={'mass':self.config_dict['dark_matter_mass']}, 
                              )
        print("Finished Setup.")
        

        
    def test_priors(self):

        sigpriorvals = np.squeeze(self.DM_prior.construct_prior_array())
        ccrbkgpriorvals = np.squeeze(self.ccr_bkg_prior.construct_prior_array())
        diffuse_astro_bkg_prior_vals = np.squeeze(self.diffuse_astro_bkg_prior.construct_prior_array())
        point_astro_bkg_prior_vals = np.squeeze(self.point_astro_bkg_prior.construct_prior_array())

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
                       logspace_riemann(point_astro_bkg_prior_vals, x=self.energy_true_axis, axis=0).T )
        
        plt.subplot(428)
        plt.plot(self.energy_true_axis, 
                       iterate_logspace_integration(diffuse_astro_bkg_prior_vals, 
                                                    axes=[self.longitudeaxistrue, 
                                                          self.latitudeaxistrue], axisindices=(1,2)).T )
        plt.xscale('log')   

        plt.show(block=self.blockplot)
        
    def simulate(self):
        print("Simulating true values...")

        self.sig_energy_vals,self.siglonvals,self.siglatvals  = self.DM_prior.sample(self.nsig)
        
        self.ccr_bkg_energy_vals,self.ccr_bkglonvals,self.ccr_bkglatvals  = self.ccr_bkg_prior.sample(self.nccr)

        self.diffuse_astro_bkg_energy_vals,self.diffuse_astro_bkglonvals,self.diffuse_astro_bkglatvals  = self.diffuse_astro_bkg_prior.sample(self.ndiffuse)

        self.point_astro_bkg_energy_vals,self.point_astro_bkglonvals,self.point_astro_bkglatvals  = self.point_astro_bkg_prior.sample(self.npoint)

        if self.diagnostics:

            plt.figure(dpi=250)

            plt.subplot(421)
            plt.hist2d(self.siglonvals,self.siglatvals, bins=(bin_centres_to_edges(self.longitudeaxistrue), bin_centres_to_edges(self.latitudeaxistrue)))


            plt.subplot(422)
            plt.hist(self.sig_energy_vals, bins=bin_centres_to_edges(self.energy_true_axis))
            plt.loglog()


            plt.subplot(423)
            plt.hist2d(self.ccr_bkglonvals,self.ccr_bkglatvals, bins=(bin_centres_to_edges(self.longitudeaxistrue), bin_centres_to_edges(self.latitudeaxistrue)))

            plt.subplot(424)
            plt.hist(self.ccr_bkg_energy_vals, bins=bin_centres_to_edges(self.energy_true_axis))
            plt.loglog()

            plt.subplot(425)
            plt.hist2d(self.diffuse_astro_bkglonvals,self.diffuse_astro_bkglatvals, bins=(bin_centres_to_edges(self.longitudeaxistrue), bin_centres_to_edges(self.latitudeaxistrue)))

            plt.subplot(426)
            plt.hist(self.diffuse_astro_bkg_energy_vals, bins=bin_centres_to_edges(self.energy_true_axis))
            plt.loglog()

            plt.subplot(427)
            plt.hist2d(self.point_astro_bkglonvals,self.point_astro_bkglatvals, bins=(bin_centres_to_edges(self.longitudeaxistrue), bin_centres_to_edges(self.latitudeaxistrue)))

            plt.subplot(428)
            plt.hist(self.point_astro_bkg_energy_vals, bins=bin_centres_to_edges(self.energy_true_axis))
            plt.loglog()

        
            plt.show(block=self.blockplot)

        print("Simulating reconstructed values...")

        self.sig_energy_meas, self.sig_longitude_meas, self.sig_latitude_meas = np.asarray([self.irf_loglike.sample(dependentvalues=[*nuisance_vals]) for nuisance_vals in tqdm(zip(self.sig_energy_vals,
                                                                                                                                                                                    self.siglonvals,
                                                                                                                                                                                    self.siglatvals))]).T
        self.ccr_bkg_energy_meas, self.ccr_bkg_longitude_meas, self.ccr_bkg_latitude_meas = np.asarray([self.irf_loglike.sample(dependentvalues=[*nuisance_vals]) for nuisance_vals in tqdm(zip(self.ccr_bkg_energy_vals,
                                                                                                                                                                                                self.ccr_bkglonvals,
                                                                                                                                                                                                self.ccr_bkglatvals))]).T
        self.diffuse_bkg_energy_meas, self.diffuse_bkg_longitude_meas, self.diffuse_bkg_latitude_meas = np.asarray([self.irf_loglike.sample(dependentvalues=[*nuisance_vals]) for nuisance_vals in tqdm(zip(self.diffuse_astro_bkg_energy_vals, 
                                                                                                                                                                                                            self.diffuse_astro_bkglonvals,
                                                                                                                                                                                                            self.diffuse_astro_bkglatvals))]).T
        self.point_bkg_energy_meas, self.point_astro_bkg_longitude_meas, self.point_astro_bkg_latitude_meas = np.asarray([self.irf_loglike.sample(dependentvalues=[*nuisance_vals]) for nuisance_vals in tqdm(zip(self.point_astro_bkg_energy_vals,
                                                                                                                                                                                                                    self.point_astro_bkglonvals,
                                                                                                                                                                                                                    self.point_astro_bkglatvals))]).T


        if self.diagnostics:
            plt.figure(dpi=250)
            plt.subplot(421)
            plt.hist2d(self.sig_longitude_meas, self.sig_latitude_meas, bins=(bin_centres_to_edges(self.longitudeaxis), bin_centres_to_edges(self.latitudeaxis)))

            plt.subplot(422)
            plt.hist(self.sig_energy_meas, bins=bin_centres_to_edges(self.energy_recon_axis))
            plt.loglog()

            plt.subplot(423)
            plt.hist2d(self.ccr_bkg_longitude_meas, self.ccr_bkg_latitude_meas, bins=(bin_centres_to_edges(self.longitudeaxis), bin_centres_to_edges(self.latitudeaxis)))

            plt.subplot(424)
            plt.hist(self.ccr_bkg_energy_meas, bins=bin_centres_to_edges(self.energy_recon_axis))
            plt.loglog()


            plt.subplot(425)
            plt.hist2d(self.diffuse_bkg_longitude_meas, self.diffuse_bkg_latitude_meas, bins=(bin_centres_to_edges(self.longitudeaxis), bin_centres_to_edges(self.latitudeaxis)))

            plt.subplot(426)
            plt.hist(self.diffuse_bkg_energy_meas, bins=bin_centres_to_edges(self.energy_recon_axis))
            plt.loglog()

            plt.subplot(427)
            plt.hist2d(self.point_astro_bkg_longitude_meas, self.point_astro_bkg_latitude_meas, bins=(bin_centres_to_edges(self.longitudeaxis), bin_centres_to_edges(self.latitudeaxis)))

            plt.subplot(428)
            plt.hist(self.point_bkg_energy_meas,  bins=bin_centres_to_edges(self.energy_recon_axis))
            plt.loglog()

        
            plt.show(block=self.blockplot)

        print("Combining reconstructed values for later use...")

        self.measured_energy = list(self.sig_energy_meas)+list(self.ccr_bkg_energy_meas)+list(self.diffuse_bkg_energy_meas)+list(self.point_bkg_energy_meas)
        self.measured_longitude = list(self.sig_longitude_meas)+list(self.ccr_bkg_longitude_meas)+list(self.diffuse_bkg_longitude_meas)+list(self.point_astro_bkg_longitude_meas)
        self.measured_latitude = list(self.sig_latitude_meas)+list(self.ccr_bkg_latitude_meas)+list(self.diffuse_bkg_latitude_meas)+list(self.point_astro_bkg_latitude_meas)
        
    def nuisance_marg(self):

        self.hyperparameter_likelihood_instance = self.discrete_hyperparameter_likelihood(
            priors                  = (self.DM_prior, self.ccr_bkg_prior, self.diffuse_astro_bkg_prior, self.point_astro_bkg_prior), 
            likelihood              = self.irf_loglike, 
            dependent_axes          = (self.energy_true_axis,  self.longitudeaxistrue, self.latitudeaxistrue), 
            hyperparameter_axes     = [*self.source_hyperparameter_input.values()], 
            numcores                = self.config_dict['numcores'], 
            likelihoodnormalisation = self.irf_norm_matrix)

        self.margresults = self.hyperparameter_likelihood_instance.nuisance_log_marginalisation(
            axisvals= (self.measured_energy, self.measured_longitude, self.measured_latitude)
            )
        


    def generate_hyper_param_likelihood(self):
        self.sigfrac_of_total_range         = construct_parameter_axes(self.config_dict['mixture_fraction_specifications']['sig/total'])
        self.ccrfrac_of_bkg_range           = construct_parameter_axes(self.config_dict['mixture_fraction_specifications']['ccr/bkg'])
        self.diffusefrac_of_astro_range     = construct_parameter_axes(self.config_dict['mixture_fraction_specifications']['diffuse/astro'])

        mixtureaxes = self.sigfrac_of_total_range, self.ccrfrac_of_bkg_range, self.diffusefrac_of_astro_range

        # self.log_hyper_param_likelihood =  0
        # for _skip_idx in tqdm(range(int(float(self.config_dict['Nevents']/self.mixture_scanning_buffer)))):
        #     _temp_log_marg_reults = [self.margresults[_index][_skip_idx*self.mixture_scanning_buffer:_skip_idx*self.mixture_scanning_buffer+self.mixture_scanning_buffer, ...] for _index in range(len(self.margresults))]
        #     self.log_hyper_param_likelihood = self.hyperparameter_likelihood_instance.update_hyperparameter_likelihood(self.hyperparameter_likelihood_instance.create_discrete_mixture_log_hyper_likelihood(
        #         mixture_axes=mixtureaxes, log_margresults=_temp_log_marg_reults))
            

        self.log_hyper_param_likelihood = self.hyperparameter_likelihood_instance.create_discrete_mixture_log_hyper_likelihood(
            mixture_axes=mixtureaxes, log_margresults=self.margresults, num_in_mixture_batch=self.mixture_scanning_buffer)

        self.log_hyper_param_likelihood=np.squeeze(self.log_hyper_param_likelihood)

        try:
            plot_log_hyper_param_likelihood = self.log_hyper_param_likelihood-special.logsumexp(self.log_hyper_param_likelihood)

            fig, ax = logdensity_matrix_plot([*mixtureaxes, *extract_axes(self.source_hyperparameter_input).values(), ], plot_log_hyper_param_likelihood, 
                                             truevals=[self.config_dict['signalfraction'], 
                                                        self.config_dict['ccr_of_bkg_fraction'], 
                                                        self.config_dict['diffuse_of_astro_fraction'],
                                                        self.config_dict['dark_matter_mass'],
                                                        0.17],
                                            sigmalines_1d=1, contours2d=1, plot_density=1, single_dim_yscales='linear',
                                            axis_names=['sig/total', 'ccr/bkg', 'diffuse/astro', 'mass [TeV]', 'alpha'], 
                                            suptitle=self.config_dict['Nevents'], figsize=(20,18))
            fig.figure.dpi = 120
            ax[3,0].set_yscale('log')
            ax[3,1].set_yscale('log')
            ax[3,2].set_yscale('log')

            ax[3,3].set_xscale('log')

            ax[4,0].set_yscale('log')
            ax[4,1].set_yscale('log')
            ax[4,2].set_yscale('log')
            ax[4,3].set_yscale('log')
            ax[4,3].set_xscale('log')
            ax[4,4].set_xscale('log')
            plt.tight_layout()
            print(f"{self.save_path}hyper_loglike_corner.pdf")
            plt.savefig(f"{self.save_path}hyper_loglike_corner.pdf")
            plt.close()
        except Exception as excpt:
            try:
                print(f"An error occurred when trying to plot results: {excpt}")

                plot_log_hyper_param_likelihood = self.log_hyper_param_likelihood-special.logsumexp(self.log_hyper_param_likelihood)

                fig, ax = logdensity_matrix_plot([*mixtureaxes, *extract_axes(self.source_hyperparameter_input).values(), ], 
                                                 plot_log_hyper_param_likelihood, 
                                                 truevals=[self.config_dict['signalfraction'], 
                                                        self.config_dict['ccr_of_bkg_fraction'], 
                                                        self.config_dict['diffuse_of_astro_fraction'],
                                                        self.config_dict['dark_matter_mass'],
                                                        0.17],
                                                sigmalines_1d=0, contours2d=0, plot_density=0, single_dim_yscales='linear',
                                                suptitle=self.config_dict['Nevents'], figsize=(20,18))
                fig.figure.dpi = 120
                ax[3,0].set_yscale('log')
                ax[3,1].set_yscale('log')
                ax[3,2].set_yscale('log')

                ax[3,3].set_xscale('log')

                ax[4,0].set_yscale('log')
                ax[4,1].set_yscale('log')
                ax[4,2].set_yscale('log')
                ax[4,3].set_yscale('log')
                ax[4,3].set_xscale('log')
                ax[4,4].set_xscale('log')

                plt.tight_layout()
                print(f"{self.save_path}hyper_loglike_corner.pdf")
                plt.savefig(f"{self.save_path}hyper_loglike_corner.pdf")
                plt.close()
            except Exception as exct:
                print(f"Could not plot results: {exct}")
            

    def run(self):

        if self.diagnostics:
            self.test_priors()

        _t1 = time.perf_counter()

        self.simulate()

        _t2 = time.perf_counter()

        self.nuisance_marg()

        _t3 = time.perf_counter()

        self.generate_hyper_param_likelihood()

        _t4 = time.perf_counter()

        print(f"Time to simulate events: {round(_t2-_t1, 3)} seconds")
        print(f"Time to marginalise over nuisance parameters: {round(_t3-_t2, 3)} seconds")
        print(f"Time to generate hyper param log-likelihood: {round(_t4-_t3, 3)} seconds")

        # self.apply_priors()


    def save(self, filename=None, pack_kwargs = {}, save_kwargs={}):
        if filename is None:
            filename = self.save_path+'run_data.pkl'

        packed_data = self.hyperparameter_likelihood_instance.pack_data(**pack_kwargs)
        packed_data['measured_event_data'] = self.measured_energy, self.measured_longitude, self.measured_latitude
        packed_data['true_event_data'] = {'sig':[self.sig_energy_vals,self.siglonvals,self.siglatvals],
                                          'ccr':[self.ccr_bkg_energy_vals,self.ccr_bkglonvals,self.ccr_bkglatvals],
                                          'diffuse': [self.diffuse_astro_bkg_energy_vals,self.diffuse_astro_bkglonvals,self.diffuse_astro_bkglatvals],
                                          'point':[self.point_bkg_energy_meas, self.point_astro_bkg_longitude_meas, self.point_astro_bkg_latitude_meas]}
        packed_data['config'] = self.config_dict
        packed_data['source_hyperparameter_input'] = self.source_hyperparameter_input

        try:
            save_to_pickle(object_to_save=packed_data, filename=filename)
        except FileNotFoundError as fnfe:
            warnings.warn("An error occured when trying to save results. Will atempt to save result to working directory as 'results.pkl'.")
            save_to_pickle(object_to_save=packed_data, filename='results.pkl')

            print("Error message: {fnfe}")


################
## For users you can copy paste the below into a stand-alone script to run this class


# if __name__=="__main__":
#     try:
#         config_file_path = sys.argv[1]
#     except:
#         warnings.warn('No configuration file given')
#         config_file_path = os.path.dirname(__file__)+'/Z2_DM_3COMP_BKG_config_default.yaml'
#     config_dict = read_config_file(config_file_path)
#     print(f"initial config_file_path: {config_file_path}")
#     standard_3COMP_BKG_instance = standard_3COMP_BKG(config_dict=config_dict,)
#     standard_3COMP_BKG_instance.run()
#     print("\n\nRun Done. Now saving")
#     standard_3COMP_BKG_instance.save()
#     print("\n\nResults saved.")
