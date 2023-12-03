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
from gammabayes.priors.astro_sources import construct_hess_source_map, construct_fermi_gaggero_matrix
from gammabayes.utils import bin_centres_to_edges
from gammabayes.utils.config_utils import create_true_axes_from_config, create_recon_axes_from_config
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
random.seed(1)

from gammabayes.likelihoods.irfs import irf_norm_setup

# log_psf_normalisations, log_edisp_normalisations = irf_norm_setup(save_results=True)


class Z2_DM_2COMP_BKG(object):
        
    def __init__(self, config_dict: dict, diagnostics: bool = False, blockplot: bool = False, stochastic=False):

        self.diagnostics    = diagnostics
        self.blockplot      = blockplot
        self.config_dict    = config_dict
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
        self.astrophysicalbackground+= construct_fermi_gaggero_matrix(energy_axis=self.energy_true_axis,
                                                                 longitudeaxis=self.longitudeaxistrue,
                                                                 latitudeaxis=self.latitudeaxistrue)


        log_astro_bkg = interpolate.RegularGridInterpolator(
            (self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue), 
            self.astrophysicalbackground
            )
        log_astro_bkgfunc = lambda energy, longitude, latitude: np.log(log_astro_bkg((energy, longitude, latitude)))


        self.astro_bkg_prior = discrete_logprior(logfunction=log_astro_bkgfunc, name='Astrophysical Background Prior',
                                    axes=(self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue,), 
                                    axes_names=['energy', 'lon', 'lat'], )
        
        self.ccr_bkg_prior = discrete_logprior(logfunction=log_bkg_CCR_dist, name='CCR Mis-identification Background Prior',
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
        astrobkgpriorvals = np.squeeze(self.astro_bkg_prior.construct_prior_array())
        ccrbkgpriorvals = np.squeeze(self.ccr_bkg_prior.construct_prior_array())

        plt.figure()
        plt.subplot(321)
        plt.title('Signal')
        plt.pcolormesh(self.longitudeaxistrue, 
                       self.latitudeaxistrue, 
                       logspace_riemann(sigpriorvals, x=self.energy_true_axis, axis=0).T )
        
        plt.subplot(322)
        plt.plot(self.energy_true_axis, 
                       iterate_logspace_integration(sigpriorvals, 
                                                    axes=[self.longitudeaxistrue, 
                                                          self.latitudeaxistrue], axisindices=(1,2)).T )
        plt.axvline(self.config_dict['mass'])
        plt.xscale('log')
        
        plt.subplot(323)
        plt.title('CCR BKG')

        plt.pcolormesh(self.longitudeaxistrue, 
                       self.latitudeaxistrue, 
                       logspace_riemann(ccrbkgpriorvals, x=self.energy_true_axis, axis=0).T )
        
        plt.subplot(324)
        plt.plot(self.energy_true_axis, 
                       iterate_logspace_integration(ccrbkgpriorvals, 
                                                    axes=[self.longitudeaxistrue, 
                                                          self.latitudeaxistrue], axisindices=(1,2)).T )
        plt.xscale('log')
        plt.subplot(325)
        plt.title('Astro BKG')
        plt.pcolormesh(self.longitudeaxistrue, 
                       self.latitudeaxistrue, 
                       logspace_riemann(astrobkgpriorvals, x=self.energy_true_axis, axis=0).T )
        
        plt.subplot(326)
        plt.plot(self.energy_true_axis, 
                       iterate_logspace_integration(astrobkgpriorvals, 
                                                    axes=[self.longitudeaxistrue, 
                                                          self.latitudeaxistrue], axisindices=(1,2)).T )
        plt.xscale('log')        

        plt.show(block=self.blockplot)
        
    def simulate(self):
        self.sig_energy_vals,self.siglonvals,self.siglatvals  = self.DM_prior.sample(
                int(round(self.config_dict['signalfraction']*self.config_dict['Nevents'])))
        
        self.ccr_bkg_energy_vals,self.ccr_bkglonvals,self.ccr_bkglatvals  = self.ccr_bkg_prior.sample(
                int(round((1-self.config_dict['signalfraction'])*self.config_dict['ccr_of_bkg_fraction']*self.config_dict['Nevents'])))

        self.astro_bkg_energy_vals,self.astro_bkglonvals,self.astro_bkglatvals  = self.astro_bkg_prior.sample(
                int(round((1-self.config_dict['signalfraction'])*(1-self.config_dict['ccr_of_bkg_fraction'])*self.config_dict['Nevents'])))
        
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

        self.sig_energy_measured = [np.squeeze(self.edisp_like.sample((energy_val,*coord,), numsamples=1)) for energy_val,coord  in notebook_tqdm(zip(self.sig_energy_vals, np.array([self.siglonvals, self.siglatvals]).T), 
                                                                                                                                                  total=int(self.config_dict['signalfraction']*self.config_dict['Nevents']))]

        self.signal_lon_measured = []
        self.signal_lat_measured = []

            
        sig_lonlat_psf_samples =  [self.psf_like.sample((energy_val,*coord,), 1).tolist() for energy_val,coord  in notebook_tqdm(zip(self.sig_energy_vals, np.array([self.siglonvals, self.siglatvals]).T), 
                                                                                                                                 total= int(round((1-self.config_dict['signalfraction'])*self.config_dict['Nevents'])))]
                
        for sig_lonlat_psf_sample in sig_lonlat_psf_samples:
            self.signal_lon_measured.append(sig_lonlat_psf_sample[0])
            self.signal_lat_measured.append(sig_lonlat_psf_sample[1])
            
        self.ccr_bkg_energy_measured = [np.squeeze(self.edisp_like.sample((energy,*coord,), numsamples=1)) for energy,coord  in notebook_tqdm(zip(self.ccr_bkg_energy_vals, np.array([self.ccr_bkglonvals, self.ccr_bkglatvals]).T), 
                                                                                                                                              total= int(round((1-self.config_dict['signalfraction'])*self.config_dict['ccr_of_bkg_fraction']*self.config_dict['Nevents'])))]

        self.ccr_bkg_lon_measured = []
        self.ccr_bkg_lat_measured = []

            
        ccr_bkg_lonlat_psf_samples =  [self.psf_like.sample((energy,*coord,), 1).tolist() for energy,coord  in notebook_tqdm(zip(self.ccr_bkg_energy_vals, np.array([self.ccr_bkglonvals, self.ccr_bkglatvals]).T), 
                                                                                                                             total= int(round((1-self.config_dict['signalfraction'])*self.config_dict['ccr_of_bkg_fraction']*self.config_dict['Nevents'])))]

        for ccr_bkg_lonlat_psf_sample in ccr_bkg_lonlat_psf_samples:
            self.ccr_bkg_lon_measured.append(ccr_bkg_lonlat_psf_sample[0])
            self.ccr_bkg_lat_measured.append(ccr_bkg_lonlat_psf_sample[1])



        self.astro_bkg_energy_measured = [np.squeeze(self.edisp_like.sample((energy,*coord,), numsamples=1)) for energy,coord  in notebook_tqdm(zip(self.astro_bkg_energy_vals, np.array([self.astro_bkglonvals, self.astro_bkglatvals]).T), 
                                                                                                                                                total= int(round((1-self.config_dict['signalfraction'])*(1-self.config_dict['ccr_of_bkg_fraction'])*self.config_dict['Nevents'])))]

        self.astro_bkg_lon_measured = []
        self.astro_bkg_lat_measured = []

            
        astro_bkg_lonlat_psf_samples =  [self.psf_like.sample((energy,*coord,), 1).tolist() for energy,coord  in notebook_tqdm(zip(self.astro_bkg_energy_vals, np.array([self.astro_bkglonvals, self.astro_bkglatvals]).T), 
                                                                                                                               total= int(round((1-self.config_dict['signalfraction'])*(1-self.config_dict['ccr_of_bkg_fraction'])*self.config_dict['Nevents'])))]

        for astro_bkg_lonlat_psf_sample in astro_bkg_lonlat_psf_samples:
            self.astro_bkg_lon_measured.append(astro_bkg_lonlat_psf_sample[0])
            self.astro_bkg_lat_measured.append(astro_bkg_lonlat_psf_sample[1])

        if self.diagnostics:
            plt.figure(dpi=250)
            plt.subplot(321)
            plt.hist2d(self.signal_lon_measured,self.signal_lat_measured, bins=(bin_centres_to_edges(self.longitudeaxis), bin_centres_to_edges(self.latitudeaxis)))

            plt.subplot(322)
            plt.hist(self.sig_energy_measured, bins=bin_centres_to_edges(self.energy_recon_axis))
            plt.loglog()

            plt.subplot(323)
            plt.hist2d(self.ccr_bkg_lon_measured, self.ccr_bkg_lat_measured, bins=(bin_centres_to_edges(self.longitudeaxis), bin_centres_to_edges(self.latitudeaxistrue)))

            plt.subplot(324)
            plt.hist(self.ccr_bkg_energy_measured, bins=bin_centres_to_edges(self.energy_recon_axis))
            plt.loglog()


            plt.subplot(323)
            plt.hist2d(self.astro_bkg_lon_measured, self.astro_bkg_lat_measured, bins=(bin_centres_to_edges(self.longitudeaxis), bin_centres_to_edges(self.latitudeaxistrue)))

            plt.subplot(324)
            plt.hist(self.astro_bkg_energy_measured, bins=bin_centres_to_edges(self.energy_recon_axis))
            plt.loglog()

        
            plt.show(block=self.blockplot)

        self.measured_energy = list(self.sig_energy_measured)+list(self.ccr_bkg_energy_measured)+list(self.astro_bkg_energy_measured)
        self.measured_lon = list(self.signal_lon_measured)+list(self.ccr_bkg_lon_measured)+list(self.astro_bkg_lon_measured)
        self.measured_lat = list(self.signal_lat_measured)+list(self.ccr_bkg_lat_measured)+list(self.astro_bkg_lat_measured)

    def nuisance_marg(self):
        self.massrange            = np.logspace(np.log10(self.energy_true_axis[1]), 2, self.config_dict['nbins_mass']) 


        self.hyperparameter_likelihood_instance = discrete_hyperparameter_likelihood(
            priors                  = (self.DM_prior, self.ccr_bkg_prior, self.astro_bkg_prior), 
            likelihood              = single_loglikelihood, 
            dependent_axes          = (self.energy_true_axis,  self.longitudeaxistrue, self.latitudeaxistrue), 
            hyperparameter_axes     = [[self.massrange], [None]], 
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
        self.sigfrac_of_total_range     = np.linspace(self.config_dict['sigfrac_lower_bd'], self.config_dict['sigfrac_upper_bd'], self.config_dict['nbins_signalfraction']) 
        self.ccrfrac_of_bkg_range       = np.linspace(self.config_dict['ccrfrac_lower_bd'], self.config_dict['ccrfrac_upper_bd'] , self.config_dict['nbins_ccrfraction'])


        new_log_posterior = self.hyperparameter_likelihood_instance.create_discrete_mixture_log_hyper_likelihood(
            mixture_axes=(self.lambdarange,), log_margresults=self.margresultsarray)

        self.log_posterior=np.squeeze(new_log_posterior)

        log_posterior = self.log_posterior-special.logsumexp(self.log_posterior)

        fig, ax = logdensity_matrix_plot([self.lambdarange, self.massrange, ], log_posterior, truevals=[self.config_dict['signalfraction'], self.config_dict['mass'],],       
                                    sigmalines_1d=0, contours2d=0, plot_density=0, single_dim_yscales='linear',
                                axis_names=['signal fraction', 'mass [TeV]',], suptitle=self.config_dict['Nevents'])
        fig.figure.dpi = 120
        ax[1,1].set_xscale('log')
        ax[1,0].set_yscale('log')
        plt.tight_layout()
        plt.show()

    def run(self):
        if self.diagnostics:
            self.test_priors()

        self.simulate()

        self.nuisance_marg()
        self.generate_hyper_param_likelihood()

if __name__=="__main__":
    config_inputs = read_config_file(os.path.dirname(__file__)+'/Z2_DM_2COMP_BKG_config_default.yaml')

    Z2_DM_2COMP_BKG_instance = Z2_DM_2COMP_BKG(config_dict=config_inputs,)
    Z2_DM_2COMP_BKG_instance.run()