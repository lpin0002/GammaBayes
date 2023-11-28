import os, sys, time
from tqdm import tqdm

from gammabayes.likelihoods.irfs import irf_loglikelihood
from gammabayes.utils.plotting import logdensity_matrix_plot
from gammabayes.utils.config_utils import read_config_file
from gammabayes.utils import logspace_riemann

from gammabayes.hyper_inference import discrete_hyperparameter_likelihood
from gammabayes.priors import discrete_logprior, log_bkg_CCR_dist
from gammabayes.likelihoods import discrete_loglike
from gammabayes.dark_matter import SS_DM_dist
from gammabayes.priors.astro_sources import construct_hess_source_map, construct_fermi_gaggero_matrix
from gammabayes.utils import bin_centres_to_edges
from gammabayes.utils.config_utils import create_true_axes_from_config, create_recon_axes_from_config

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
from gammabayes.likelihoods.irfs import irf_norm_setup



class Z2_DM_single_comp_BKG(object):

    def __init__(self, config_dict, plots=True, plotblock=False):
        self.config_dict = config_dict

        self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue   = create_true_axes_from_config(self.config_dict)
        self.energy_recon_axis, self.longitudeaxis, self.latitudeaxis          = create_recon_axes_from_config(self.config_dict)

        self.plots = plots
        self.plotblock = plotblock

        self.log_psf_normalisations, self.log_edisp_normalisations = irf_norm_setup(energy_true_axis=self.energy_true_axis, 
                                                                        longitudeaxistrue=self.longitudeaxistrue,
                                                                        latitudeaxistrue=self.latitudeaxistrue,
                                                                        energy_recon_axis=self.energy_recon_axis,
                                                                        longitudeaxis=self.longitudeaxis,
                                                                        latitudeaxis=self.latitudeaxis)


        self.astrophysicalbackground =   construct_hess_source_map(energy_true_axis         = self.energy_true_axis, 
                                                                   longitudeaxistrue        = self.longitudeaxistrue, 
                                                                   latitudeaxistrue         = self.latitudeaxistrue)
        self.astrophysicalbackground +=   construct_fermi_gaggero_matrix(energy_true_axis   = self.energy_true_axis, 
                                                                   longitudeaxistrue        = self.longitudeaxistrue, 
                                                                   latitudeaxistrue         = self.latitudeaxistrue)
        if self.plots:
            plt.figure()
            plt.pcolormesh(self.longitudeaxistrue, self.latitudeaxistrue, np.mean(self.astrophysicalbackground, axis=0).T)
            plt.xlabel('Longitude [deg]')
            plt.ylabel('Latitude [deg]')
            plt.show(block=self.plotblock)


        energymeshtrue, lonmeshtrue, latmeshtrue    = np.meshgrid(self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue, indexing='ij')
        lonmeshrecon, latmeshrecon                  = np.meshgrid(self.longitudeaxis, self.latitudeaxis, indexing='ij')


        unnormed_logbkgpriorvalues = np.logaddexp(np.squeeze(log_bkg_CCR_dist(energymeshtrue, lonmeshtrue,latmeshtrue)),np.log(self.astrophysicalbackground))

        # Not required, but possibly will make the normalisation more accurate
        logbkgpriorvalues = unnormed_logbkgpriorvalues - logspace_riemann(
            logy=logspace_riemann(
                logy=logspace_riemann(
                    logy=unnormed_logbkgpriorvalues, x=self.energy_true_axis, axis=0),
                    x = self.longitudeaxistrue, axis=0),
                    x = self.latitudeaxistrue, axis=0)

        logbkgfunc_interp= interpolate.RegularGridInterpolator(
            (self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue), 
            np.exp(logbkgpriorvalues)
            )
        self.logbkgfunc = lambda energy, longitude, latitude: np.log(logbkgfunc_interp((energy, longitude, latitude)))


        self.bkg_prior = discrete_logprior(logfunction=self.logbkgfunc, name='Background Prior',
                                    axes=[self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue,], 
                                    axes_names=['energy', 'lon', 'lat'],)


        SS_DM_dist_instance= SS_DM_dist(self.longitudeaxistrue, self.latitudeaxistrue, density_profile=self.config_dict['dmdensity_profile'])
        self.logDMpriorfunc = SS_DM_dist_instance.func_setup()




        self.DM_prior = discrete_logprior(logfunction=self.logDMpriorfunc, name='Scalar Singlet Dark Matter Prior',
                                    axes=(self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue,), 
                                    axes_names=['energy', 'lon', 'lat'],
                                    default_hyperparameter_values=[self.config_dict['mass']], 
                                    hyperparameter_names=['mass'], )


        self.irf_loglikelihood_instance = irf_loglikelihood(axes=[self.energy_recon_axis, self.longitudeaxis, self.latitudeaxis], 
                                    dependent_axes=[self.energy_true_axis, self.longitudeaxistrue, self.latitudeaxistrue],)
    
    def simulate(self):
        ################################################################################################
        ################################################################################################
        ################## 'True' Value Simulations
        self.sig_energy_vals,self.siglonvals,self.siglatvals  = self.DM_prior.sample(
                int(round(self.config_dict['signalfraction']*self.config_dict['Nevents'])))

        self.bkg_energy_vals,self.bkglonvals,self.bkglatvals  = self.bkg_prior.sample(
                int(round((1-self.config_dict['signalfraction'])*self.config_dict['Nevents'])))


        ################################################################################################
        ################################################################################################
        ################## 'Measured'/Reconstructed Value Simulations
        self.sig_energy_measured,self.signal_lon_measured,self.signal_lat_measured  = np.asarray(
            [np.squeeze(self.irf_loglikelihood_instance.sample((energy_val,*coord,), numsamples=1)) for energy_val,coord  in tqdm(zip(self.sig_energy_vals, 
                                                                                                                                      np.array([self.siglonvals, self.siglatvals]).T), 
                                                                                                                                      total=int(self.config_dict['signalfraction']*self.config_dict['Nevents']))]).T

        self.bkg_energy_measured, self.bkg_lon_measured, self.bkg_lat_measured = np.asarray(
            [np.squeeze(self.irf_loglikelihood_instance.sample((energy_val,*coord,), numsamples=1)) for energy_val,coord  in tqdm(zip(self.bkg_energy_vals, 
                                                                                                                              np.array([self.bkglonvals, self.bkglatvals]).T), 
                                                                                                                              total=(1-self.config_dict['signalfraction'])*self.config_dict['Nevents'])]).T

        if self.plots:
            plt.figure()
            plt.subplot(2,2,1)
            plt.title('SIG')
            plt.hist2d(self.signal_lon_measured,self.signal_lat_measured, bins=[bin_centres_to_edges(self.longitudeaxistrue), bin_centres_to_edges(self.latitudeaxistrue)])
            plt.xlabel('Longitude [deg]')
            plt.ylabel('Latitude [deg]')

            plt.subplot(2,2,2)
            plt.hist2d(self.sig_energy_measured, bins=bin_centres_to_edges(self.energy_true_axis), )
            plt.xlabel('Energy [TeV]')


            plt.subplot(2,2,3)
            plt.title('BKG')

            plt.hist2d(self.bkg_lon_measured, self.bkg_lat_measured, bins=[bin_centres_to_edges(self.longitudeaxistrue), bin_centres_to_edges(self.latitudeaxistrue)])
            plt.xlabel('Longitude [deg]')
            plt.ylabel('Latitude [deg]')

            plt.subplot(2,2,4)
            plt.hist2d(self.bkg_energy_measured, bins=bin_centres_to_edges(self.energy_true_axis), )
            plt.xlabel('Energy [TeV]')

            plt.show(block=self.plotblock)
        
        ################################################################################################
        ################################################################################################
        ################## Combining Simulations
        self.measured_energy = list(self.sig_energy_measured)+list(self.bkg_energy_measured)
        self.measured_lon = list(self.signal_lon_measured)+list(self.bkg_lon_measured)
        self.measured_lat = list(self.signal_lat_measured)+list(self.bkg_lat_measured)

    def nuisance_marginalisation(self):

        #### Setting up range of mass values to test
        print(f"Mass range; min: {self.energy_true_axis.min()}, max: {2}")
        self.massrange            = np.logspace(np.log10(self.energy_true_axis.min()), 2, self.config_dict['nbins_mass']) 

        ################################################################################################
        ################################################################################################
        ################## Nuisance Parameter Marginalisation

        self.hyperparameter_likelihood_instance = discrete_hyperparameter_likelihood(
            priors              = (self.DM_prior, self.bkg_prior,), 
            likelihood          = self.irf_loglikelihood_instance, 
            dependent_axes      = (self.energy_true_axis,  self.longitudeaxistrue, self.latitudeaxistrue), 
            hyperparameter_axes = [[self.massrange], [None]], 
            numcores            = self.config_dict['numcores'], 
            likelihoodnormalisation = self.log_psf_normalisations+self.log_edisp_normalisations)

        margresults = self.hyperparameter_likelihood_instance.nuisance_log_marginalisation(
            axisvals= (self.measured_energy, self.measured_lon, self.measured_lat)
            )

        self.margresultsarray = np.array(margresults)

    def generate_hyperparameter_likelihood(self):
        try:
            if self.config_dict['sigfrac_spacing']=='log':
                self.sigfrac_range            = np.logspace(np.log10(self.config_dict['sigfrac_lower_bd']), 
                                                    np.log10(self.config_dict['sigfrac_upper_bd']), 
                                                    self.config_dict['nbins_signalfraction'])
            elif self.config_dict['sigfrac_spacing']=='linear':
                self.sigfrac_range            = np.linspace(self.config_dict['sigfrac_lower_bd'], 
                                                    self.config_dict['sigfrac_upper_bd'], 
                                                    self.config_dict['nbins_signalfraction'])
            else:
                warnings.warn("Spacing for signal fraction values incorrect. Must be 'linear', or 'log'. Defaulting to linear.")
                self.sigfrac_range            = np.linspace(self.config_dict['sigfrac_lower_bd'], 
                                                    self.config_dict['sigfrac_upper_bd'], 
                                                    self.config_dict['nbins_signalfraction'])
        except:
            warnings.warn("'sigfrac_spacing' values not given in config. Must be 'linear', or 'log'. Defaulting to linear.")
            self.sigfrac_range            = np.linspace(self.config_dict['sigfrac_lower_bd'], 
                                                self.config_dict['sigfrac_upper_bd'], 
                                                self.config_dict['nbins_signalfraction'])


        log_hyperparam_likelihood = self.hyperparameter_likelihood_instance.create_discrete_mixture_log_hyper_likelihood(
            mixture_axes=(self.sigfrac_range,), log_margresults=self.margresultsarray)


        self.log_hyperparam_likelihood=np.squeeze(log_hyperparam_likelihood)

        plt.figure()
        plt.pcolormesh(np.squeeze(log_hyperparam_likelihood))
        plt.show()

    def apply_hyper_priors(self):
        # self.log_posterior, _, _ = self.hyperparameter_likelihood_instance.apply_uniform_hyperparameter_priors(
        #                                                                                 priorinfos=({'spacing':'linear','uniformity':'linear'}, # Signal fraction axis
        #                                                                                              {'spacing':'log10','uniformity':'log'}),   # Mass axis
        #                                                                                 hyper_param_axes=[self.sigfrac_range, self.massrange, ])
        self.log_posterior = self.log_hyperparam_likelihood

    def run(self):
        self.simulate()
        self.nuisance_marginalisation()
        self.generate_hyperparameter_likelihood()
        self.apply_hyper_priors()


    def plot(self):
        self.log_posterior = self.log_posterior - logsumexp(self.log_posterior)
        fig, ax = logdensity_matrix_plot([self.sigfrac_range, self.massrange, ], 
                                        self.log_hyperparam_likelihood, 
                                        truevals=[self.config_dict['signalfraction'], self.config_dict['mass'],],
                                        sigmalines_1d=0, contours2d=0, 
                                        plot_density=0, single_dim_yscales='linear',
                                        axis_names=['signal fraction', 'mass [TeV]',],
                                        suptitle=self.config_dict['Nevents'])
        fig.figure.dpi = 120
        ax[1,1].set_xscale('log')
        ax[1,0].set_yscale('log')
        fig.show()
        
        return fig, ax
    

if __name__=="__main__":
    from os import path
    config_inputs = read_config_file(path.dirname(__file__)+'/Z2_DM_1COMP_BKG_config_default.yaml')

    Z2_DM_single_comp_BKG_instance = Z2_DM_single_comp_BKG(config_inputs)

    Z2_DM_single_comp_BKG_instance.run()

    Z2_DM_single_comp_BKG_instance.plot()
    plt.show()

