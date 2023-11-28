
import os, sys, time
from tqdm import tqdm

from gammabayes.likelihoods.irfs import log_edisp, log_psf, single_loglikelihood, log_psf_normalisations, log_edisp_normalisations
from gammabayes.utils.plotting import logdensity_matrix_plot
from gammabayes.utils.event_axes import energy_true_axis, longitudeaxistrue, latitudeaxistrue, energy_recon_axis, longitudeaxis, latitudeaxis, makelogjacob
from gammabayes.samplers import discrete_hyperparameter_continuous_mix_post_process_sampler

from gammabayes.hyper_inference import discrete_hyperparameter_likelihood
from gammabayes.priors import discrete_logprior, log_bkg_CCR_dist
from gammabayes.likelihoods.likelihood import discrete_loglikelihood
from gammabayes.dark_matter import SS_DM_dist
from gammabayes.priors.astro_sources import construct_hess_source_map_interpolation, construct_log_fermi_gaggero_bkg

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from tqdm.autonotebook import tqdm as notebook_tqdm

import random, pandas as pd
random.seed(0)


numberoftruevaluesamples = int(1e3)
Nsamples=numberoftruevaluesamples
sigfraction                 = 0.5
ccr_of_bkg_fraction         = 0.9
diffuse_of_astro_fraction   = 0.2
nsig                        = int(round(sigfraction*Nsamples))
nastrodiffuse               = int(round((1-sigfraction)*(1-ccr_of_bkg_fraction)*diffuse_of_astro_fraction*Nsamples))
nastropoint                 =int(round((1-sigfraction)*(1-ccr_of_bkg_fraction)*(1-diffuse_of_astro_fraction)*Nsamples))
nccr                        = int(round((1-sigfraction)*ccr_of_bkg_fraction*Nsamples))
print(nsig, nccr, nastrodiffuse, nastropoint,)
if nccr+nastrodiffuse+nastropoint+nsig!=numberoftruevaluesamples:
    print(1/0)
numcores                    = 8
truemass                    = 50.0


true_logjacob = np.meshgrid(makelogjacob(energy_true_axis),longitudeaxistrue, latitudeaxistrue, indexing='ij')[0]


diffuse_astro_bkg_prior = discrete_logprior(logfunction=construct_log_fermi_gaggero_bkg(), name='Diffuse Astrophysical Background Prior',
                               axes=(energy_true_axis, longitudeaxistrue, latitudeaxistrue,), 
                               axes_names=['energy', 'lon', 'lat'], logjacob=true_logjacob)


point_astro_bkg_prior = discrete_logprior(logfunction=construct_hess_source_map_interpolation(), name='Point Source Astrophysical Background Prior',
                               axes=(energy_true_axis, longitudeaxistrue, latitudeaxistrue,), 
                               axes_names=['energy', 'lon', 'lat'], logjacob=true_logjacob)


ccr_bkg_prior = discrete_logprior(logfunction=log_bkg_CCR_dist, name='CCR Mis-identification Background Prior',
                               axes=(energy_true_axis, longitudeaxistrue, latitudeaxistrue,), 
                               axes_names=['energy', 'lon', 'lat'], logjacob=true_logjacob)


SS_DM_dist_instance= SS_DM_dist(longitudeaxistrue, latitudeaxistrue)
logDMpriorfunc = SS_DM_dist_instance.func_setup()

DM_prior = discrete_logprior(logfunction=logDMpriorfunc, name='Scalar Singlet Dark Matter Prior',
                               axes=(energy_true_axis, longitudeaxistrue, latitudeaxistrue,), axes_names=['energy', 'lon', 'lat'],
                               default_hyperparameter_values=(truemass,), hyperparameter_names=['mass'], logjacob=true_logjacob)
DM_prior

sig_energy_vals,siglonvals,siglatvals                                   = DM_prior.sample(nsig)
ccr_energy_vals,ccrlonvals,ccrlatvals                                   = ccr_bkg_prior.sample(nccr)
diffuse_astro_energy_vals,diffuse_astrolonvals,diffuse_astrolatvals     = diffuse_astro_bkg_prior.sample(nastrodiffuse)
point_astro_energy_vals,point_astrolonvals,point_astrolatvals           = point_astro_bkg_prior.sample(nastropoint)



logjacob = np.log(energy_recon_axis)
edisp_like = discrete_loglikelihood(logfunction=log_edisp, 
                                    axes=(energy_recon_axis,), axes_names='log10E recon',
                                    name='energy dispersion',
                                    dependent_axes=(energy_true_axis, longitudeaxistrue, latitudeaxistrue,), logjacob=logjacob,
                                    dependent_axes_names = ['log10E true', 'lon', 'lat'])

psf_like = discrete_loglikelihood(logfunction=log_psf, 
                                    axes=(longitudeaxis, latitudeaxis), axes_names=['longitude recon', 'latitude recon'],
                                    name='point spread function ',
                                    dependent_axes=(energy_true_axis, longitudeaxistrue, latitudeaxistrue,),
                                    dependent_axes_names = ['log10E true', 'lon', 'lat'])

signal_lon_measured = []
signal_lat_measured = []
if nsig!=0:
    signal_energy_measured = [np.squeeze(edisp_like.sample((energy_val,*coord,), numsamples=1)) for energy_val,coord  in notebook_tqdm(zip(sig_energy_vals, np.array([siglonvals, siglatvals]).T), total=nsig)]
    sig_lonlat_psf_samples =  [psf_like.sample((energy_val,*coord,), 1).tolist() for energy_val,coord  in notebook_tqdm(zip(sig_energy_vals, np.array([siglonvals, siglatvals]).T), total=nsig)]
    
    for sig_lonlat_psf_sample in sig_lonlat_psf_samples:
        signal_lon_measured.append(sig_lonlat_psf_sample[0])
        signal_lat_measured.append(sig_lonlat_psf_sample[1])
else:
    signal_energy_measured = np.asarray([])

diffuse_astro_lon_measured = []
diffuse_astro_lat_measured = []
if nastrodiffuse!=0:
    diffuse_astro_energy_measured = [np.squeeze(edisp_like.sample((energy_val,*coord,), numsamples=1)) for energy_val,coord  in notebook_tqdm(zip(diffuse_astro_energy_vals, np.array([diffuse_astrolonvals, diffuse_astrolatvals]).T), total=nastrodiffuse)]
    diffuse_astro_lonlat_psf_samples =  [psf_like.sample((logeval,*coord,), 1).tolist() for logeval,coord  in notebook_tqdm(zip(diffuse_astro_energy_vals, np.array([diffuse_astrolonvals, diffuse_astrolatvals]).T), total=nastrodiffuse)]
    
    for diffuse_astro_lonlat_psf_sample in diffuse_astro_lonlat_psf_samples:
        diffuse_astro_lon_measured.append(diffuse_astro_lonlat_psf_sample[0])
        diffuse_astro_lat_measured.append(diffuse_astro_lonlat_psf_sample[1])
else:
    diffuse_astro_energy_measured = np.asarray([])

point_astro_lon_measured = []
point_astro_lat_measured = []
if nastrodiffuse!=0:
    point_astro_energy_measured = [np.squeeze(edisp_like.sample((energy_val,*coord,), numsamples=1)) for energy_val,coord  in notebook_tqdm(zip(point_astro_energy_vals, np.array([point_astrolonvals, point_astrolatvals]).T), total=nastropoint)]
    point_astro_lonlat_psf_samples =  [psf_like.sample((energy_val,*coord,), 1).tolist() for energy_val,coord  in notebook_tqdm(zip(point_astro_energy_vals, np.array([point_astrolonvals, point_astrolatvals]).T), total=nastropoint)]
    
    for point_astro_lonlat_psf_sample in point_astro_lonlat_psf_samples:
        point_astro_lon_measured.append(point_astro_lonlat_psf_sample[0])
        point_astro_lat_measured.append(point_astro_lonlat_psf_sample[1])
else:
    point_astro_energy_measured = np.asarray([])

ccr_lon_measured = []
ccr_lat_measured = []
if nccr!=0:
    ccr_energy_measured = [np.squeeze(edisp_like.sample((energy_val,*coord,), numsamples=1)) for energy_val,coord  in notebook_tqdm(zip(ccr_energy_vals, np.array([ccrlonvals, ccrlatvals]).T), total=nccr)]
    ccr_lonlat_psf_samples =  [psf_like.sample((energy_val,*coord,), 1).tolist() for energy_val,coord  in notebook_tqdm(zip(ccr_energy_vals, np.array([ccrlonvals, ccrlatvals]).T), total=nccr)]
    
    for ccr_lonlat_psf_sample in ccr_lonlat_psf_samples:
        ccr_lon_measured.append(ccr_lonlat_psf_sample[0])
        ccr_lat_measured.append(ccr_lonlat_psf_sample[1])
else:
    ccr_energy_measured = np.asarray([])

measured_energy = list(signal_energy_measured)+list(diffuse_astro_energy_measured)+list(point_astro_energy_measured)+list(ccr_energy_measured)
measured_lon = list(signal_lon_measured)+list(diffuse_astro_lon_measured)+list(point_astro_lon_measured)+list(ccr_lon_measured)
measured_lat = list(signal_lat_measured)+list(diffuse_astro_lat_measured)+list(point_astro_lat_measured)+list(ccr_lat_measured)


nbins_logmass=81

if nsig is None:
    nsig = len(list(measured_energy))

logmasswindowwidth      = 20/np.sqrt(nsig)

logmasslowerbound       = np.log10(truemass)-logmasswindowwidth
logmassupperbound       = np.log10(truemass)+logmasswindowwidth

# if 1:
if logmasslowerbound<np.log10(energy_recon_axis)[0]:
    logmasslowerbound = np.log10(energy_recon_axis)[0]
# if 1:
if logmassupperbound>2:
    logmassupperbound = 2


massrange            = np.logspace(logmasslowerbound, logmassupperbound, nbins_logmass) 

hyperparameter_likelihood_instance = discrete_hyperparameter_likelihood(priors=(DM_prior, ccr_bkg_prior, diffuse_astro_bkg_prior, point_astro_bkg_prior), likelihood=single_loglikelihood, 
                                                               dependent_axes=(energy_true_axis,  longitudeaxistrue, latitudeaxistrue), dependent_logjacob=true_logjacob,
                                                               hyperparameter_axes = ((massrange,), (None,), (None,), (None, )), 
                                                               numcores=numcores, likelihoodnormalisation = log_psf_normalisations+log_edisp_normalisations)

measured_energy = [float(measured_energy_val) for measured_energy_val in measured_energy]
margresults = hyperparameter_likelihood_instance.nuisance_log_marginalisation(axisvals= (measured_energy, measured_lon, measured_lat))

margresultsarray = np.array(margresults)

nbins_sigfrac               = 51
sigfrac_windowwidth         = 10/np.sqrt(Nsamples)
ccrfrac_windowwidth         = 8/np.sqrt(Nsamples)
diffusefrac_windowwidth     = 16/np.sqrt(Nsamples)

nbins_ccr_frac              = 51
nbins_diffuse_frac          = 51

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

diffusefrac_of_astro_range_upperbound = diffuse_of_astro_fraction+diffusefrac_windowwidth
diffusefrac_of_astro_range_lowerbound = diffuse_of_astro_fraction-diffusefrac_windowwidth


if diffusefrac_of_astro_range_upperbound>1:
    diffusefrac_of_astro_range_upperbound = 1
if diffusefrac_of_astro_range_lowerbound<0:
    diffusefrac_of_astro_range_lowerbound = 0


sigfracrange = np.linspace(sigfrac_range_lowerbound,    sigfrac_range_upperbound,   nbins_sigfrac)
ccrfrac_of_bkg_range = np.linspace(ccrfrac_of_bkg_range_lowerbound,ccrfrac_of_bkg_range_upperbound , nbins_ccr_frac)
diffuse_of_astro_range = np.linspace(diffusefrac_of_astro_range_lowerbound,diffusefrac_of_astro_range_upperbound, nbins_diffuse_frac)


skipfactor = 10
mixtureaxes = sigfracrange, ccrfrac_of_bkg_range, diffuse_of_astro_range
new_log_posterior = 0
loopstart = time.perf_counter()
for dataidx in tqdm(range(int(round(margresultsarray.shape[0]/skipfactor)))):
    tempmargresultsarray = margresultsarray[dataidx*skipfactor:dataidx*skipfactor+skipfactor]
    new_log_posterior += hyperparameter_likelihood_instance.create_discrete_mixture_log_hyper_likelihood(
        mixture_axes=(*mixtureaxes,), log_margresults=tempmargresultsarray)

    

new_log_posterior = new_log_posterior - special.logsumexp(new_log_posterior)

endertimer = time.perf_counter()


log_posterior = np.squeeze(new_log_posterior)

from gammabayes.utils.plotting import logdensity_matrix_plot
from scipy.stats import norm as norm1d
fig, ax = logdensity_matrix_plot(axes=(*mixtureaxes, massrange,), logprobmatrix=log_posterior-special.logsumexp(log_posterior), 
                       truevals=(sigfraction, ccr_of_bkg_fraction, diffuse_of_astro_fraction, truemass,),
                       axis_names=('signal fraction', 'ccr of bkg fraction', 'diffuse of astro fraction', r'mass [TeV]', ),
                       dpi=140, figsize=(12,10))

[ax.set_yscale('log') for ax in ax[-1,:][:-1]];
ax[-1,-1].set_xscale('log')
plt.show()

discrete_hyperparameter_continuous_mix_sampler_instance = discrete_hyperparameter_continuous_mix_post_process_sampler(
    hyper_param_ranges_tuple=((massrange,), (None,), (None,), (None,),), mixture_axes=mixtureaxes,
    margresultsarray  = margresultsarray,
    nestedsampler_kwarg_dict ={'nlive':1000}
    )

posterior_results = discrete_hyperparameter_continuous_mix_sampler_instance.generate_log_hyperlike(
    run_nested_kwarg_dict = {'dlogz':0.5},
    )

posterior_results_equal = posterior_results.samples_equal()

from corner import corner

def title_format_func(title, value, uncertainty, fmt='.2e'):
    exponent = int(fmt[-1])  # Extract the exponent from the format string
    formatted_value = format(value, fmt)
    formatted_uncertainty = format(uncertainty, fmt)
    return f'{title} = ({formatted_value} +/- {formatted_uncertainty}) * 10^{exponent}'


defaults_kwargs = dict(
    bins=64,
    smooth=1.5, 
    smooth1d=1.0,
    label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16), title_fmt='.3f', color='#0072C1',
    truth_color='tab:orange', 
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.), 1 - np.exp(-8), 1 - np.exp(-25/2)),
    plot_datapoints=True, 
    fill_contours=True,
    max_n_ticks=4, 
    )



ranges = [(min(posterior_results_equal[:, i])-0.5*np.ptp(posterior_results_equal[:, i]), 
           max(posterior_results_equal[:, i])+0.5*np.ptp(posterior_results_equal[:, i]),
           ) for i in range(posterior_results_equal.shape[1])]


for idx, range_idx in enumerate(ranges):
    if not(min(posterior_results_equal[:, idx])<0):
        if range_idx[0]<0 and range_idx[1]>1:
            ranges[idx] = (0,1)
        elif range_idx[0]<0:
            ranges[idx] = (0,range_idx[1])
        else:
            ranges[idx] = (range_idx[0], 1)




fig = corner(posterior_results_equal,# range=ranges,
    labels=['sig/total', 'ccr/bkg','diffuse/astro bkg', r'$m_\chi$ [TeV]'],
     show_titles=True, truths =(sigfraction, ccr_of_bkg_fraction, diffuse_of_astro_fraction,truemass),  **defaults_kwargs)
axes = fig.axes
plt.suptitle(f"Nevents = {Nsamples}", size=24)
plt.tight_layout()
plt.savefig(time.strftime(f'Figures/testplot_mass({truemass})_mixfracs{sigfraction, ccr_of_bkg_fraction, diffuse_of_astro_fraction}_%m_%d_%H:%M.pdf'))
plt.show()

