
import os, sys, time
from tqdm import tqdm

# from gammabayes.BFCalc.createspectragrids import singlechannel_diffflux, getspectrafunc, darkmatterdoubleinput, energymassinputspectralfunc
from gammabayes.likelihoods.irfs import log_edisp, log_psf, single_loglikelihood, log_psf_normalisations, log_edisp_normalisations
from gammabayes.utils.plotting import logdensity_matrix_plot
from gammabayes.utils.event_axes import energy_true_axis, longitudeaxistrue, latitudeaxistrue, energy_recon_axis, longitudeaxis, latitudeaxis, makelogjacob
from gammabayes.utils.config_utils import read_config_file
from gammabayes.utils import logspace_simpson
from gammabayes.samplers import discrete_hyperparameter_continuous_mix_post_process_sampler

# from gammabayes.utils.utils import inverse_transform_sampling
from gammabayes.hyper_inference import discrete_hyperparameter_likelihood
from gammabayes.priors import discrete_logprior, log_bkg_CCR_dist
from gammabayes.likelihoods.likelihood import discrete_loglikelihood
from gammabayes.dark_matter import SS_DM_dist
from gammabayes.priors.astro_sources import construct_hess_source_map, construct_fermi_gaggero_matrix
from gammabayes.utils.utils import bin_centres_to_edges

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
from corner import corner

random.seed(0)


astrophysicalbackground = construct_hess_source_map()+construct_fermi_gaggero_matrix()

energy_meshtrue, lonmeshtrue, latmeshtrue = np.meshgrid(energy_true_axis, longitudeaxistrue, latitudeaxistrue, indexing='ij')
lonmeshrecon, latmeshrecon = np.meshgrid(longitudeaxis, latitudeaxis, indexing='ij')

logjacobtrue = np.meshgrid(np.log(energy_meshtrue), longitudeaxistrue, latitudeaxistrue, indexing='ij')[0]



numberoftruevaluesamples = int(1e3)
Nsamples=numberoftruevaluesamples
sigfraction                 = 0.5
ccr_of_bkg_fraction         = 0.8
nsig                        = int(round(sigfraction*Nsamples))
nastro                      = int(round((1-sigfraction)*(1-ccr_of_bkg_fraction)*Nsamples))
nccr                        = int(round((1-sigfraction)*ccr_of_bkg_fraction*Nsamples))
if nccr+nastro+nsig!=numberoftruevaluesamples:
    print(1/0)
numcores            = 8
truemass            = 1.0


unnormed_log_astro_bkgpriorvalues = np.log(astrophysicalbackground)


log_astro_bkgpriorvalues = unnormed_log_astro_bkgpriorvalues - logspace_simpson(
    logy=logspace_simpson(
        logy=logspace_simpson(
            logy=unnormed_log_astro_bkgpriorvalues, x=energy_true_axis, axis=0),
            x = longitudeaxistrue, axis=0),
            x = latitudeaxistrue, axis=0)



log_astro_bkg = interpolate.RegularGridInterpolator(
    (energy_true_axis, longitudeaxistrue, latitudeaxistrue), 
    np.exp(log_astro_bkgpriorvalues)
    )
log_astro_bkgfunc = lambda energy, longitude, latitude: np.log(log_astro_bkg((energy, longitude, latitude)))


astro_bkg_prior = discrete_logprior(logfunction=log_astro_bkgfunc, name='Astrophysical Background Prior',
                               axes=(energy_true_axis, longitudeaxistrue, latitudeaxistrue,), 
                               axes_names=['energy', 'lon', 'lat'], 
                               logjacob=np.meshgrid(makelogjacob(energy_true_axis),longitudeaxistrue, latitudeaxistrue, indexing='ij')[0])

ccr_bkg_prior = discrete_logprior(logfunction=log_bkg_CCR_dist, name='CCR Mis-identification Background Prior',
                               axes=(energy_true_axis, longitudeaxistrue, latitudeaxistrue,), 
                               axes_names=['energy', 'lon', 'lat'], 
                               logjacob=np.meshgrid(makelogjacob(energy_true_axis),longitudeaxistrue, latitudeaxistrue, indexing='ij')[0])


SS_DM_dist_instance= SS_DM_dist(longitudeaxistrue, latitudeaxistrue)
logDMpriorfunc = SS_DM_dist_instance.func_setup()

DM_prior = discrete_logprior(logfunction=logDMpriorfunc, name='Scalar Singlet Dark Matter Prior',
                               axes=(energy_true_axis, longitudeaxistrue, latitudeaxistrue,), 
                               axes_names=['energy', 'lon', 'lat'],
                               default_hyperparameter_values=[truemass], 
                               hyperparameter_names=['mass'], 
                               logjacob=np.meshgrid(makelogjacob(energy_true_axis),longitudeaxistrue, latitudeaxistrue, indexing='ij')[0])

sig_energy_vals,siglonvals,siglatvals  = DM_prior.sample(nsig)

ccr_energy_vals,ccrlonvals,ccrlatvals  = ccr_bkg_prior.sample(nccr)

astro_energy_vals,astrolonvals,astrolatvals  = astro_bkg_prior.sample(nastro)


logjacob = np.meshgrid(np.log(energy_recon_axis), longitudeaxis, latitudeaxis, indexing='ij')[0]

logjacob = np.log(energy_recon_axis)
edisp_like = discrete_loglikelihood(logfunction=log_edisp, 
                                    axes=(energy_recon_axis,), axes_names='log10E recon',
                                    name='energy dispersion',
                                    dependent_axes=(energy_true_axis, longitudeaxistrue, latitudeaxistrue,), logjacob=logjacob,
                                    dependent_axes_names = ['E true', 'lon', 'lat'])

psf_like = discrete_loglikelihood(logfunction=log_psf, 
                                    axes=(longitudeaxis, latitudeaxis), axes_names=['longitude recon', 'latitude recon'],
                                    name='point spread function ',
                                    dependent_axes=(energy_true_axis, longitudeaxistrue, latitudeaxistrue,),
                                    dependent_axes_names = ['E true', 'lon', 'lat'])
psf_like

signal_lon_measured = []
signal_lat_measured = []
signal_energy_measured = [np.squeeze(edisp_like.sample((energy_val,*coord,), numsamples=1)) for energy_val,coord  in notebook_tqdm(zip(sig_energy_vals, np.array([siglonvals, siglatvals]).T), total=nsig)]
sig_lonlat_psf_samples =  [psf_like.sample((energy_val,*coord,), 1).tolist() for energy_val,coord  in notebook_tqdm(zip(sig_energy_vals, np.array([siglonvals, siglatvals]).T), total=nsig)]

for sig_lonlat_psf_sample in sig_lonlat_psf_samples:
    signal_lon_measured.append(sig_lonlat_psf_sample[0])
    signal_lat_measured.append(sig_lonlat_psf_sample[1])


astro_lon_measured = []
astro_lat_measured = []
astro_energy_measured = [np.squeeze(edisp_like.sample((energy_val,*coord,), numsamples=1)) for energy_val,coord  in notebook_tqdm(zip(astro_energy_vals, np.array([astrolonvals, astrolatvals]).T), total=nastro)]
astro_lonlat_psf_samples =  [psf_like.sample((energy_val,*coord,), 1).tolist() for energy_val,coord  in notebook_tqdm(zip(astro_energy_vals, np.array([astrolonvals, astrolatvals]).T), total=nastro)]

for astro_lonlat_psf_sample in astro_lonlat_psf_samples:
    astro_lon_measured.append(astro_lonlat_psf_sample[0])
    astro_lat_measured.append(astro_lonlat_psf_sample[1])


ccr_lon_measured = []
ccr_lat_measured = []

ccr_energy_measured = [np.squeeze(edisp_like.sample((energy_val,*coord,), numsamples=1)) for energy_val,coord  in notebook_tqdm(zip(ccr_energy_vals, np.array([ccrlonvals, ccrlatvals]).T), total=nccr)]
ccr_lonlat_psf_samples =  [psf_like.sample((energy_val,*coord,), 1).tolist() for energy_val,coord  in notebook_tqdm(zip(ccr_energy_vals, np.array([ccrlonvals, ccrlatvals]).T), total=nccr)]

for ccr_lonlat_psf_sample in ccr_lonlat_psf_samples:
    ccr_lon_measured.append(ccr_lonlat_psf_sample[0])
    ccr_lat_measured.append(ccr_lonlat_psf_sample[1])


measured_energy = list(signal_energy_measured)+list(astro_energy_measured)+list(ccr_energy_measured)
measured_lon = list(signal_lon_measured)+list(astro_lon_measured)+list(ccr_lon_measured)
measured_lat = list(signal_lat_measured)+list(astro_lat_measured)+list(ccr_lat_measured)


nbinsmass=81

if nsig is None:
    nsig = len(list(measured_energy))

logmasswindowwidth      = 20/np.sqrt(nsig)

logmasslowerbound       = np.log10(truemass)-logmasswindowwidth
logmassupperbound       = np.log10(truemass)+logmasswindowwidth

# if 1:
if logmasslowerbound<np.log10(energy_true_axis[0]):
    logmasslowerbound = np.log10(energy_true_axis[0])
# if 1:
if logmassupperbound>2:
    logmassupperbound = 2


massrange            = np.logspace(logmasslowerbound, logmassupperbound, nbinsmass) 


hyperparameter_likelihood_instance = discrete_hyperparameter_likelihood(priors=(DM_prior, ccr_bkg_prior, astro_bkg_prior,), 
                                                                likelihood=single_loglikelihood, 
                                                               dependent_axes=(energy_true_axis,  longitudeaxistrue, latitudeaxistrue), 
                                                               hyperparameter_axes = ((massrange,), (None,), (None,),), 
                                                               numcores=numcores, 
                                                               likelihoodnormalisation = log_psf_normalisations+log_edisp_normalisations)

measured_energy = [float(measured_energy_val) for measured_energy_val in measured_energy]
margresults = hyperparameter_likelihood_instance.nuisance_log_marginalisation(axisvals= (measured_energy, measured_lon, measured_lat))

margresultsarray = np.array(margresults)

nbins_sigfrac               = 121
sigfrac_windowwidth         = 8/np.sqrt(Nsamples)
ccrfrac_windowwidth         = 6/np.sqrt(Nsamples)

nbins_ccr_frac              = 111

sigfrac_range_upperbound = sigfraction+sigfrac_windowwidth
sigfrac_range_lowerbound = sigfraction-sigfrac_windowwidth



if sigfrac_range_upperbound>1:
    sigfrac_range_upperbound = 1
if sigfrac_range_lowerbound<0:
    sigfrac_range_lowerbound = 0

ccrfrac_of_bkg_range_upperbound = ccr_of_bkg_fraction+ccrfrac_windowwidth
ccrfrac_of_bkg_range_lowerbound = ccr_of_bkg_fraction-ccrfrac_windowwidth

if ccrfrac_of_bkg_range_upperbound>1:
    ccrfrac_of_bkg_range_upperbound = 1
if ccrfrac_of_bkg_range_lowerbound<0:
    ccrfrac_of_bkg_range_lowerbound = 0


sigfracrange = np.linspace(sigfrac_range_lowerbound,    sigfrac_range_upperbound,   nbins_sigfrac)
ccrfrac_of_bkg_range = np.linspace(ccrfrac_of_bkg_range_lowerbound,ccrfrac_of_bkg_range_upperbound , nbins_ccr_frac)


skipfactor = 10
mixtureaxes = sigfracrange, ccrfrac_of_bkg_range


discrete_hyperparameter_continuous_mix_sampler_instance = discrete_hyperparameter_continuous_mix_post_process_sampler(
    hyper_param_ranges_tuple=((massrange,), (None,), (None,)), mixture_axes=mixtureaxes, margresultsarray  = margresultsarray,
    nestedsampler_kwarg_dict ={'nlive':900}, numcores=10
    )

posterior_results = discrete_hyperparameter_continuous_mix_sampler_instance.generate_log_hyperlike(
    run_nested_kwarg_dict = {'dlogz':0.1, 'n_effective':10000}, 
    )

posterior_results_equal = posterior_results.samples_equal()

defaults_kwargs = dict(
    bins=50, 
    smooth=2.0, 
    smooth1d=1.0,
    label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16), color='#0072C1',
    truth_color='tab:orange', 
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.), 1 - np.exp(-8)),
    plot_density=True, 
    plot_datapoints=True, 
    fill_contours=True,
    max_n_ticks=4, 
    )

corner(posterior_results_equal, 
    labels=['sig/total', 'ccr/bkg', r'$m_\chi$ [TeV]'],
     show_titles=True, truths =(sigfraction, ccr_of_bkg_fraction, truemass),  **defaults_kwargs)
plt.suptitle(f"Nevents: {Nsamples}", size=24)

plt.tight_layout()
plt.show()



