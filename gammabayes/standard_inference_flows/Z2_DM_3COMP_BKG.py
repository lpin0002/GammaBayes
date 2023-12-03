import os, sys, time
from tqdm import tqdm

from gammabayes.likelihoods.irfs import log_edisp, log_psf, single_loglikelihood
from gammabayes.utils.plotting import logdensity_matrix_plot
from gammabayes.utils.config_utils import read_config_file
from gammabayes.utils import logspace_riemann, iterate_logspace_integration
from gammabayes.hyper_inference import discrete_hyperparameter_likelihood
from gammabayes.priors import discrete_logprior, log_bkg_CCR_dist
from gammabayes.likelihoods import discrete_loglike
from gammabayes.dark_matter import SS_DM_dist
from gammabayes.priors.astro_sources import construct_hess_source_map, construct_fermi_gaggero_matrix, construct_hess_source_map_interpolation, construct_log_fermi_gaggero_bkg
from gammabayes.utils import bin_centres_to_edges
from gammabayes.utils.config_utils import create_true_axes_from_config, create_recon_axes_from_config
from gammabayes.samplers import discrete_hyperparameter_continuous_mix_post_process_sampler

# from gammabayes.utils.event_axes import energy_true_axis, longitudeaxistrue, latitudeaxistrue, energy_recon_axis, longitudeaxis, latitudeaxis
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from astropy import units as u
from scipy import interpolate, special, integrate, stats
from scipy.integrate import simps
from matplotlib import cm
from tqdm.autonotebook import tqdm as notebook_tqdm

import functools, random
from multiprocessing import Pool, freeze_support
import multiprocessing
import pandas as pd
# random.seed(1)

from gammabayes.likelihoods.irfs import irf_norm_setup

# log_psf_normalisations, log_edisp_normalisations = irf_norm_setup(save_results=True)


class Z2_DM_3COMP_BKG(object):
        
    def __init__(self, config_dict: dict = {}, config_file_path = None):
        try:
            self.config_dict    = config_dict
            assert (len(self.config_dict)>0) and (type(self.config_dict)==dict)
        except:
            try:
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
            self.name    = self.config_dict['name']
        except:
            self.name    = time.strftime(f"DM{self.config_dict['mass']}_SIGFRAC{self.config_dict['signalfraction']}_CCRFRAC{self.config_dict['ccr_of_bkg_fraction']}_DIFFUSEFRAC{self.config_dict['diffuse_of_astro_fraction']}_3COMP_BKG_%m|%d_%H:%M")

        try:
            self.save_path = self.config_dict['save_path']
        except:
            self.save_path = f'data/{self.name}'
            os.makedirs('data', exist_ok=True)
            os.makedirs(f'data/{self.name}', exist_ok=True)


        try:
            self.save_figure_path = self.config_dict['save_figure_path']
            self.save_eventdata_path = self.config_dict['save_eventdata_path']
            self.save_analysisresults_path = self.config_dict['save_analysisresults_path']
        except:
            self.save_figure_path = self.save_path
            self.save_eventdata_path = self.save_path
            self.save_analysisresults_path = self.save_path

        os.makedirs(self.save_figure_path, exist_ok=True)
        os.makedirs(self.save_eventdata_path, exist_ok=True)
        os.makedirs(self.save_analysisresults_path, exist_ok=True)


        self.nsig                        = int(round(self.config_dict['signalfraction']*self.config_dict['Nevents']))
        self.nccr                        = int(round((1-self.config_dict['signalfraction'])*self.config_dict['ccr_of_bkg_fraction']*self.config_dict['Nevents']))
        self.ndiffuse               = int(round((1-self.config_dict['signalfraction'])*(1-self.config_dict['ccr_of_bkg_fraction'])*self.config_dict['diffuse_of_astro_fraction']*self.config_dict['Nevents']))
        self.npoint                 =int(round((1-self.config_dict['signalfraction'])*(1-self.config_dict['ccr_of_bkg_fraction'])*(1-self.config_dict['diffuse_of_astro_fraction'])*self.config_dict['Nevents']))


        self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue   = create_true_axes_from_config(self.config_dict)
        self.energy_recon_axis, self.longitudeaxis, self.latitudeaxis          = create_recon_axes_from_config(self.config_dict)


        self.log_psf_normalisations, self.log_edisp_normalisations = irf_norm_setup(energy_true_axis=self.energy_true_axis, 
                                                                                    energy_recon_axis=self.energy_recon_axis, 
                                                                                    longitudeaxistrue=self.longitudeaxistrue, 
                                                                                    longitudeaxis=self.longitudeaxis, 
                                                                                    latitudeaxistrue=self.latitudeaxistrue, 
                                                                                    latitudeaxis=self.latitudeaxis)


        self.astrophysicalbackground = construct_hess_source_map(energy_axis=self.energy_true_axis,
                                                                 longitudeaxis=self.longitudeaxistrue,
                                                                 latitudeaxis=self.latitudeaxistrue)
        
        self.ccr_bkg_prior = discrete_logprior(logfunction=log_bkg_CCR_dist, name='CCR Mis-identification Background Prior',
                               axes=(self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue,), 
                                    axes_names=['energy', 'lon', 'lat'], )

        self.diffuse_astro_bkg_prior = discrete_logprior(logfunction=construct_log_fermi_gaggero_bkg(), name='Diffuse Astrophysical Background Prior',
                               axes=(self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue,), 
                                    axes_names=['energy', 'lon', 'lat'], )
        
        self.point_astro_bkg_prior = discrete_logprior(logfunction=construct_hess_source_map_interpolation(), name='Point Source Astrophysical Background Prior',
                               axes=(self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue,), 
                                    axes_names=['energy', 'lon', 'lat'], )

        self.SS_DM_dist_instance= SS_DM_dist(self.longitudeaxistrue, self.latitudeaxistrue, density_profile=self.config_dict['dmdensity_profile'])
        self.logDMpriorfunc = self.SS_DM_dist_instance.func_setup()

        self.DM_prior = discrete_logprior(logfunction=self.logDMpriorfunc, name='Scalar Singlet Dark Matter Prior',
                                    axes=(self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue,), 
                                    axes_names=['energy', 'lon', 'lat'],
                                    default_hyperparameter_values=[self.config_dict['mass']], 
                                    hyperparameter_names=['mass'], )
        
        self.edisp_like = discrete_loglike(logfunction=log_edisp, 
                                    axes=(self.energy_recon_axis,), axes_names='E recon',
                                    name='energy dispersion',
                                    dependent_axes=(self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue,), 
                                    dependent_axes_names = ['E true', 'lon true', 'lat true'])

        self.psf_like = discrete_loglike(logfunction=log_psf, 
                                            axes=(self.longitudeaxis, self.latitudeaxis), axes_names=['longitude recon', 'latitude recon'],
                                            name='point spread function ', 
                                            dependent_axes=(self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue,),
                                            dependent_axes_names = ['E true', 'lon', 'lat'])
        
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
        plt.axvline(self.config_dict['mass'])
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
                       logspace_riemann(diffuse_astro_bkg_prior_vals, x=self.energy_true_axis, axis=0).T )
        
        plt.subplot(428)
        plt.plot(self.energy_true_axis, 
                       iterate_logspace_integration(diffuse_astro_bkg_prior_vals, 
                                                    axes=[self.longitudeaxistrue, 
                                                          self.latitudeaxistrue], axisindices=(1,2)).T )
        plt.xscale('log')   

        plt.show(block=self.blockplot)
        
    def simulate(self):
        self.sig_energy_vals,self.siglonvals,self.siglatvals  = self.DM_prior.sample(self.nsig)
        
        self.ccr_bkg_energy_vals,self.ccr_bkglonvals,self.ccr_bkglatvals  = self.ccr_bkg_prior.sample(self.nccr)

        self.diffuse_astro_bkg_energy_vals,self.diffuse_astro_bkglonvals,self.diffuse_astro_bkglatvals  = self.diffuse_astro_bkg_prior.sample(self.ndiffuse)

        self.point_astro_bkg_energy_vals,self.point_astro_bkglonvals,self.point_astro_bkglatvals  = self.point_astro_bkg_prior.sample(self.npoint)

        if self.diagnostics:

            plt.figure(dpi=250)

            plt.subplot(321)
            plt.hist2d(self.siglonvals,self.siglatvals, bins=(bin_centres_to_edges(self.longitudeaxistrue), bin_centres_to_edges(self.latitudeaxistrue)))


            plt.subplot(322)
            sigtrue_histvals = plt.hist(self.sig_energy_vals, bins=bin_centres_to_edges(self.energy_true_axis))
            logspectralvals = self.logDMpriorfunc(self.energy_true_axis, self.energy_true_axis*0.0, self.energy_true_axis*0.0, self.energy_true_axis*0.0+self.config_dict['mass'])
            spectralvals = np.exp(logspectralvals)
            plt.plot(self.energy_true_axis,spectralvals/np.mean(spectralvals)*np.mean(sigtrue_histvals[0]), lw=0.5)
            plt.loglog()


            plt.subplot(323)
            plt.hist2d(self.ccr_bkglonvals,self.ccr_bkglatvals, bins=(bin_centres_to_edges(self.longitudeaxistrue), bin_centres_to_edges(self.latitudeaxistrue)))

            plt.subplot(324)
            plt.hist(self.ccr_bkg_energy_vals, bins=bin_centres_to_edges(self.energy_true_axis))
            plt.loglog()

            plt.subplot(325)
            plt.hist2d(self.astro_bkglonvals,self.astro_bkglatvals, bins=(bin_centres_to_edges(self.longitudeaxistrue), bin_centres_to_edges(self.latitudeaxistrue)))

            plt.subplot(326)
            plt.hist(self.astro_bkg_energy_vals, bins=bin_centres_to_edges(self.energy_true_axis))
            plt.loglog()

        
            plt.show(block=self.blockplot)

        self.sig_energy_measured = [np.squeeze(self.edisp_like.sample((energy_val,*coord,), numsamples=1)) for energy_val,coord  in notebook_tqdm(zip(self.sig_energy_vals, 
                                                                                                                                                      np.array([self.siglonvals, 
                                                                                                                                                                self.siglatvals]).T), 
                                                                                                                                                  total=self.nsig)]
        self.signal_lon_measured = []
        self.signal_lat_measured = []
        sig_lonlat_psf_samples =  [self.psf_like.sample((energy_val,*coord,), 1).tolist() for energy_val,coord  in notebook_tqdm(zip(self.sig_energy_vals, 
                                                                                                                                     np.array([self.siglonvals, 
                                                                                                                                               self.siglatvals]).T), 
                                                                                                                                 total= self.nsig)]
        for sig_lonlat_psf_sample in sig_lonlat_psf_samples:
            self.signal_lon_measured.append(sig_lonlat_psf_sample[0])
            self.signal_lat_measured.append(sig_lonlat_psf_sample[1])


            
        self.ccr_bkg_energy_measured = [np.squeeze(self.edisp_like.sample((energy,*coord,), numsamples=1)) for energy,coord  in notebook_tqdm(zip(self.ccr_bkg_energy_vals, 
                                                                                                                                                  np.array([self.ccr_bkglonvals, 
                                                                                                                                                            self.ccr_bkglatvals]).T), 
                                                                                                                                              total= self.nccr)]
        self.ccr_bkg_lon_measured = []
        self.ccr_bkg_lat_measured = []
        ccr_bkg_lonlat_psf_samples =  [self.psf_like.sample((energy,*coord,), 1).tolist() for energy,coord  in notebook_tqdm(zip(self.ccr_bkg_energy_vals, 
                                                                                                                                 np.array([self.ccr_bkglonvals, 
                                                                                                                                           self.ccr_bkglatvals]).T), 
                                                                                                                             total= self.nccr)]
        for ccr_bkg_lonlat_psf_sample in ccr_bkg_lonlat_psf_samples:
            self.ccr_bkg_lon_measured.append(ccr_bkg_lonlat_psf_sample[0])
            self.ccr_bkg_lat_measured.append(ccr_bkg_lonlat_psf_sample[1])



        self.diffuse_astro_bkg_energy_measured = [np.squeeze(self.edisp_like.sample((energy,*coord,), numsamples=1)) for energy,coord  in notebook_tqdm(zip(self.diffuse_astro_bkg_energy_vals, 
                                                                                                                                                            np.array([self.diffuse_astro_bkglonvals, 
                                                                                                                                                                      self.diffuse_astro_bkglatvals]).T), 
                                                                                                                                                total= self.ndiffuse)]
        self.diffuse_astro_bkg_lon_measured = []
        self.diffuse_astro_bkg_lat_measured = []
        diffuse_astro_bkg_lonlat_psf_samples =  [self.psf_like.sample((energy,*coord,), 1).tolist() for energy,coord  in notebook_tqdm(zip(self.diffuse_astro_bkg_energy_vals, 
                                                                                                                                           np.array([self.diffuse_astro_bkglonvals, 
                                                                                                                                                     self.diffuse_astro_bkglatvals]).T), 
                                                                                                                               total= self.ndiffuse)]

        for diffuse_astro_bkg_lonlat_psf_sample in diffuse_astro_bkg_lonlat_psf_samples:
            self.diffuse_astro_bkg_lon_measured.append(diffuse_astro_bkg_lonlat_psf_sample[0])
            self.diffuse_astro_bkg_lat_measured.append(diffuse_astro_bkg_lonlat_psf_sample[1])


        self.point_astro_bkg_energy_measured = [np.squeeze(self.edisp_like.sample((energy,*coord,), numsamples=1)) for energy,coord  in notebook_tqdm(zip(self.point_astro_bkg_energy_vals, 
                                                                                                                                                          np.array([self.point_astro_bkglonvals, 
                                                                                                                                                                    self.point_astro_bkglatvals]).T), 
                                                                                                                                                total= self.npoint)]

        self.point_astro_bkg_lon_measured = []
        self.point_astro_bkg_lat_measured = []

            
        point_astro_bkg_lonlat_psf_samples =  [self.psf_like.sample((energy,*coord,), 1).tolist() for energy,coord  in notebook_tqdm(zip(self.point_astro_bkg_energy_vals, 
                                                                                                                                         np.array([self.point_astro_bkglonvals, 
                                                                                                                                                   self.point_astro_bkglatvals]).T), 
                                                                                                                               total= self.npoint)]

        for point_astro_bkg_lonlat_psf_sample in point_astro_bkg_lonlat_psf_samples:
            self.point_astro_bkg_lon_measured.append(point_astro_bkg_lonlat_psf_sample[0])
            self.point_astro_bkg_lat_measured.append(point_astro_bkg_lonlat_psf_sample[1])


        if self.diagnostics:
            plt.figure(dpi=250)
            plt.subplot(421)
            plt.hist2d(self.signal_lon_measured,self.signal_lat_measured, bins=(bin_centres_to_edges(self.longitudeaxis), bin_centres_to_edges(self.latitudeaxis)))

            plt.subplot(422)
            plt.hist(self.sig_energy_measured, bins=bin_centres_to_edges(self.energy_recon_axis))
            plt.loglog()

            plt.subplot(423)
            plt.hist2d(self.ccr_bkg_lon_measured, self.ccr_bkg_lat_measured, bins=(bin_centres_to_edges(self.longitudeaxis), bin_centres_to_edges(self.latitudeaxistrue)))

            plt.subplot(424)
            plt.hist(self.ccr_bkg_energy_measured, bins=bin_centres_to_edges(self.energy_recon_axis))
            plt.loglog()


            plt.subplot(425)
            plt.hist2d(self.diffuse_astro_bkg_lon_measured, self.diffuse_astro_bkg_lat_measured, bins=(bin_centres_to_edges(self.longitudeaxis), bin_centres_to_edges(self.latitudeaxistrue)))

            plt.subplot(426)
            plt.hist(self.diffuse_astro_bkg_energy_measured, bins=bin_centres_to_edges(self.energy_recon_axis))
            plt.loglog()

            plt.subplot(425)
            plt.hist2d(self.point_astro_bkg_lon_measured, self.point_astro_bkg_lat_measured, bins=(bin_centres_to_edges(self.longitudeaxis), bin_centres_to_edges(self.latitudeaxistrue)))

            plt.subplot(426)
            plt.hist(self.point_astro_bkg_energy_measured, bins=bin_centres_to_edges(self.energy_recon_axis))
            plt.loglog()

        
            plt.show(block=self.blockplot)

        self.measured_energy = list(self.sig_energy_measured)+list(self.ccr_bkg_energy_measured)+list(self.diffuse_astro_bkg_energy_measured)+list(self.point_astro_bkg_energy_measured)
        self.measured_lon = list(self.signal_lon_measured)+list(self.ccr_bkg_lon_measured)+list(self.diffuse_astro_bkg_lon_measured)+list(self.point_astro_bkg_lon_measured)
        self.measured_lat = list(self.signal_lat_measured)+list(self.ccr_bkg_lat_measured)+list(self.diffuse_astro_bkg_lat_measured)+list(self.point_astro_bkg_lat_measured)

    def nuisance_marg(self):
        self.massrange            = np.logspace(np.log10(self.config_dict['mass'])-5/np.sqrt(self.config_dict['Nevents']), 
                                                        np.log10(self.config_dict['mass'])+5/np.sqrt(self.config_dict['Nevents']),
                                                        self.config_dict['nbins_mass']) 

        self.hyperparameter_likelihood_instance = discrete_hyperparameter_likelihood(
            priors                  = (self.DM_prior, self.ccr_bkg_prior, self.diffuse_astro_bkg_prior, self.point_astro_bkg_prior), 
            likelihood              = single_loglikelihood, 
            dependent_axes          = (self.energy_true_axis,  self.longitudeaxistrue, self.latitudeaxistrue), 
            hyperparameter_axes     = [[self.massrange], [None], [None], [None]], 
            numcores                = self.config_dict['numcores'], 
            likelihoodnormalisation = self.log_psf_normalisations+self.log_edisp_normalisations)

        self.measured_energy = [float(measured_energy_val) for measured_energy_val in self.measured_energy]
        margresults = self.hyperparameter_likelihood_instance.nuisance_log_marginalisation(
            axisvals= (self.measured_energy, self.measured_lon, self.measured_lat)
            )

        self.margresultsarray = np.asarray(margresults)
        sigmargresults = np.squeeze(np.vstack(self.margresultsarray[:,0])).T


        from matplotlib.pyplot import get_cmap
        cmap = get_cmap(name='cool')

        if self.diagnostics:
            plt.figure()
            logmass_lines = np.exp(sigmargresults-logspace_riemann(sigmargresults, x=self.massrange, axis=0)).T
            for idx, line in enumerate(logmass_lines):
                plt.plot(self.massrange, line, label=idx, c=cmap(idx/len(logmass_lines)), alpha=0.5, lw=1.0)
            plt.axvline(self.config_dict['mass'], c='tab:orange')
            plt.xscale('log')
            plt.show(block=self.blockplot)

    def generate_hyper_param_likelihood(self):
        self.sigfrac_of_total_range         = np.linspace(self.config_dict['sigfrac_lower_bd'], self.config_dict['sigfrac_upper_bd'], self.config_dict['nbins_signalfraction']) 
        self.ccrfrac_of_bkg_range           = np.linspace(self.config_dict['ccrfrac_lower_bd'], self.config_dict['ccrfrac_upper_bd'] , self.config_dict['nbins_ccrfraction'])
        self.diffusefrac_of_astro_range     = np.linspace(self.config_dict['diffusefrac_lower_bd'], self.config_dict['diffusefrac_upper_bd'] , self.config_dict['nbins_diffusefraction'])

        mixtureaxes = self.sigfrac_of_total_range, self.ccrfrac_of_bkg_range, self.diffusefrac_of_astro_range


        if self.config_dict['mixture_fraction_sampling']=='scan':
            skipfactor = 10
            log_hyper_param_likelihood = 0
            for dataidx in tqdm(range(int(round(self.margresultsarray.shape[0]/skipfactor)))):
                tempmargresultsarray = self.margresultsarray[dataidx*skipfactor:dataidx*skipfactor+skipfactor]
                log_hyper_param_likelihood += self.hyperparameter_likelihood_instance.create_discrete_mixture_log_hyper_likelihood(
                    mixture_axes=(*mixtureaxes,), log_margresults=tempmargresultsarray)


            self.log_hyper_param_likelihood=np.squeeze(log_hyper_param_likelihood)




            plot_log_hyper_param_likelihood = self.log_hyper_param_likelihood-special.logsumexp(self.log_hyper_param_likelihood)

            fig, ax = logdensity_matrix_plot([self.sigfrac_of_total_range, self.ccrfrac_of_bkg_range, self.diffusefrac_of_astro_range, self.massrange, ], plot_log_hyper_param_likelihood, 
                                            truevals=[self.config_dict['signalfraction'], 
                                                      self.config_dict['ccr_of_bkg_fraction'], 
                                                      self.config_dict['diffuse_of_astro_fraction'], 
                                                      self.config_dict['mass'],],   
                                            sigmalines_1d=1, contours2d=1, plot_density=1, single_dim_yscales='linear',
                                            axis_names=['signal/total', 'ccr/bkg', 'diffuse/astro', 'mass [TeV]',], suptitle=self.config_dict['Nevents'])
            fig.figure.dpi = 120
            ax[3,3].set_xscale('log')
            ax[3,0].set_yscale('log')
            ax[3,1].set_yscale('log')
            ax[3,2].set_yscale('log')
            plt.tight_layout()
            plt.show()
        else:
            from corner import corner
            from gammabayes.utils.plotting import defaults_kwargs
            discrete_hyperparameter_continuous_mix_sampler_instance = discrete_hyperparameter_continuous_mix_post_process_sampler(
                hyper_param_ranges_tuple=((self.massrange,), (None,), (None,), (None,)), mixture_axes=mixtureaxes, margresultsarray  = self.margresultsarray,
                nestedsampler_kwarg_dict ={'nlive':900}, numcores=10
                )
            
            self.posterior_results = discrete_hyperparameter_continuous_mix_sampler_instance.generate_log_hyperlike(
                run_nested_kwarg_dict = {'dlogz':0.5,}, 
                )
            



            corner(np.asarray(self.posterior_results), 
                labels=['sig/total', 'ccr/bkg', r'$m_\chi$ [TeV]'],
                ranges=((0,1),(0,1),(0,1),(1e-1,1e2)),
                show_titles=True, truths =(self.config_dict['signalfraction'], 
                                           self.config_dict['ccr_of_bkg_fraction'], 
                                           self.config_dict['diffuse_of_astro_fraction'], 
                                           self.config_dict['mass']),  **defaults_kwargs)
            plt.suptitle(f"Nevents: {self.config_dict['Nevents']}", size=24)

            plt.tight_layout()
            plt.show()
            
            

    def run(self):
        if self.diagnostics:
            self.test_priors()

        self.simulate()
        
        self.nuisance_marg()
        self.generate_hyper_param_likelihood()

        if self.config_dict['mixture_fraction_sampling']=='scan':
            # self.apply_priors()

            pass

if __name__=="__main__":
    config_file_path = sys.argv[1]

    Z2_DM_3COMP_BKG_instance = Z2_DM_3COMP_BKG(config_file_path=config_file_path,)
    Z2_DM_3COMP_BKG_instance.run()