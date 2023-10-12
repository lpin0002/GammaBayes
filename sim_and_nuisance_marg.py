# $$ [markdown]
# # <h1><b><I> General Setup

# $$ [markdown]
# ## Imports and general setup

# $$
import os, sys
sys.path.append("gammabayes/dark_matter/")
sys.path.append("gammabayes")

from gammabayes.dark_matter.createspectragrids import singlechannel_diffflux, getspectrafunc, darkmatterdoubleinput, energymassinputspectralfunc
from gammabayes.utils.utils import log10eaxistrue, longitudeaxistrue, latitudeaxistrue, log10eaxis, longitudeaxis, latitudeaxis, time,psf, edisp, bkgdist, interpolate, special, integrate
from gammabayes.utils.utils import SkyCoord, WcsGeom, inverse_transform_sampling, tqdm#, setup_full_fake_signal_dist_copy_vers#, setup_full_fake_signal_dist, diff_irf_marg
from GammaBayes.gammabayes.hyper_inference.hyperparameter_likelihood import hyperparameter_likelihood
from GammaBayes.gammabayes.priors.prior import discrete_logprior
from GammaBayes.gammabayes.likelihoods.likelihood import discrete_loglikelihood
from gammabayes.utils.utils import edisp_test, psf_test, log10eaxis, longitudeaxis, latitudeaxis
from gammabayes.utils.utils import psf_efficient, edisp_efficient, edisp_test, psf_test, single_likelihood
from gammabayes.utils.load_package_data import astrophysicalbackground, psfnormalisationvalues, edispnormalisationvalues
from GammaBayes.gammabayes.dark_matter.SS_DM_Construct import SS_DM_dist


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from astropy import units as u
from scipy import special,stats
from scipy.integrate import simps
from matplotlib import cm
from tqdm.autonotebook import tqdm as notebook_tqdm

import functools, multiprocessing, yaml
from multiprocessing import Pool, freeze_support
import pandas as pd
import warnings

from gammabayes.utils.utils import read_config_file, save_config_file




# $$
log10emeshtrue, lonmeshtrue, latmeshtrue = np.meshgrid(log10eaxistrue, longitudeaxistrue, latitudeaxistrue, indexing='ij')
lonmeshrecon, latmeshrecon = np.meshgrid(longitudeaxis, latitudeaxis, indexing='ij')

logjacobtrue = np.meshgrid(np.log(10**log10eaxistrue), longitudeaxistrue, latitudeaxistrue, indexing='ij')[0]


print(lonmeshtrue.shape, lonmeshrecon.shape)

# $$ [markdown]
# ## Script Parameter Setup

# $$
startertimer = time.perf_counter()
print(startertimer)

config_inputs = read_config_file(sys.argv[1])

stemfolder = f"data/{config_inputs['identifier']}"
stemdatafolder = stemfolder+"/singlerundata"
datafolder = stemdatafolder+f"/{config_inputs['runnumber']}"

os.makedirs('data', exist_ok=True)
os.makedirs(f"data/{config_inputs['identifier']}", exist_ok=True)
os.makedirs(stemdatafolder, exist_ok=True)


if config_inputs['batch_job']:
    os.makedirs(datafolder, exist_ok=True)
else:
    os.makedirs(datafolder, exist_ok=False)




nsig                = int(round(config_inputs['xi']*config_inputs['Nevents']))
nbkg                = int(round((1-config_inputs['xi'])*config_inputs['Nevents']))

config_inputs['xi'] = nsig/(nsig+nbkg)

save_config_file(config_inputs, datafolder+'/inputconfig.yaml')


# $$ [markdown]
# # <h1><b>Simulation

# $$ [markdown]
# ## Setup

# $$ [markdown]
# ### Background setup

# $$
unnormed_logbkgpriorvalues = np.logaddexp(np.squeeze(bkgdist(log10emeshtrue, lonmeshtrue,latmeshtrue)),np.log(astrophysicalbackground))


logbkgpriorvalues = unnormed_logbkgpriorvalues - special.logsumexp(unnormed_logbkgpriorvalues+logjacobtrue)

logbkgpriorvalues.shape


nuisancemesh = np.meshgrid(log10eaxistrue, longitudeaxistrue, latitudeaxistrue, indexing='ij')

unnormed_logbkgpriorvalues = np.logaddexp(np.squeeze(bkgdist(*nuisancemesh)),np.log(astrophysicalbackground))
# unnormed_logbkgpriorvalues = np.squeeze(bkgdist(*nuisancemesh))


logbkgfunc_annoying = interpolate.RegularGridInterpolator((log10eaxistrue, longitudeaxistrue, latitudeaxistrue), np.exp(unnormed_logbkgpriorvalues))
logbkgfunc = lambda logenergy, longitude, latitude: np.log(logbkgfunc_annoying((logenergy, longitude, latitude)))


bkg_prior = discrete_logprior(logfunction=logbkgfunc, name='Background Prior',
                               axes=(log10eaxistrue, longitudeaxistrue, latitudeaxistrue,), 
                               axes_names=['energy', 'lon', 'lat'], logjacob=logjacobtrue)

# $$ [markdown]
# ### Signal Setup

# $$
SS_DM_dist_instance= SS_DM_dist(longitudeaxistrue, latitudeaxistrue)
logDMpriorfunc = SS_DM_dist_instance.func_setup()

# $$
DM_prior = discrete_logprior(logfunction=logDMpriorfunc, name='Scalar Singlet Dark Matter Prior',
                               axes=(log10eaxistrue, longitudeaxistrue, latitudeaxistrue,), axes_names=['energy', 'lon', 'lat'],
                               default_hyperparameter_values=(config_inputs['logmass'],), hyperparameter_names=['mass'], logjacob=logjacobtrue)
DM_prior

# $$ [markdown]
# ## True Value Simulation

# $$
if config_inputs['xi']!=0.0:
    siglogevals,siglonvals,siglatvals  = DM_prior.sample(nsig)
else:
    siglogevals = np.asarray([])
    siglonvals = np.asarray([])
    siglatvals = np.asarray([])

# $$

if config_inputs['xi']!=1.0:
    bkglogevals,bkglonvals,bkglatvals  = bkg_prior.sample(nbkg)
else:
    bkglogevals = np.asarray([])
    bkglonvals = np.asarray([])
    bkglatvals = np.asarray([])


np.save(f'{datafolder}/true_bkg_samples.npy', np.array([bkglogevals, bkglonvals, bkglatvals]))
np.save(f'{datafolder}/true_sig_samples.npy', np.array([siglogevals, siglonvals, siglatvals]))

# $$ [markdown]
# ## Reconstructed Value Simulation

# $$
logjacob = np.meshgrid(np.log(10**log10eaxis), longitudeaxis, latitudeaxis, indexing='ij')[0]

# $$
logjacob = np.log(10**log10eaxis)
edisp_like = discrete_loglikelihood(logfunction=edisp_test, 
                                    axes=(log10eaxis,), axes_names='log10E recon',
                                    name='energy dispersion',
                                    dependent_axes=(log10eaxistrue, longitudeaxistrue, latitudeaxistrue,), logjacob=logjacob,
                                    dependent_axes_names = ['log10E true', 'lon', 'lat'])

# $$
psf_like = discrete_loglikelihood(logfunction=psf_test, 
                                    axes=(longitudeaxis, latitudeaxis), axes_names=['longitude recon', 'latitude recon'],
                                    name='point spread function ',
                                    dependent_axes=(log10eaxistrue, longitudeaxistrue, latitudeaxistrue,),
                                    dependent_axes_names = ['log10E true', 'lon', 'lat'])
psf_like

# $$ [markdown]
# ### Signal

# $$

if config_inputs['xi']!=0.0:
    # sig_log10e_edisp_samples = [edisp_like.sample(signal_event_tuple, 1).tolist() for signal_event_tuple in tqdm(sig_samples.T)]
    signal_log10e_measured = [np.squeeze(edisp_like.sample((logeval,*coord,), numsamples=1)) for logeval,coord  in notebook_tqdm(zip(siglogevals, np.array([siglonvals, siglatvals]).T), 
                                                                                                                                 total=nsig, ncols=80)]
else:
    signal_log10e_measured = np.asarray([])

# $$

signal_lon_measured = []
signal_lat_measured = []

if config_inputs['xi']!=0:
    
    sig_lonlat_psf_samples =  [psf_like.sample((logeval,*coord,), 1).tolist() for logeval,coord  in notebook_tqdm(zip(siglogevals, np.array([siglonvals, siglatvals]).T), 
                                                                                                                  total=nsig, ncols=80)]
    
    for sig_lonlat_psf_sample in sig_lonlat_psf_samples:
        signal_lon_measured.append(sig_lonlat_psf_sample[0])
        signal_lat_measured.append(sig_lonlat_psf_sample[1])

# $$ [markdown]
# ### Background

# $$
if config_inputs['xi']!=1.0:
    # sig_log10e_edisp_samples = [edisp_like.sample(signal_event_tuple, 1).tolist() for signal_event_tuple in tqdm(sig_samples.T)]
    bkg_log10e_measured = [np.squeeze(edisp_like.sample((logeval,*coord,), numsamples=1)) for logeval,coord  in notebook_tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), 
                                                                                                                              total=nbkg, ncols=80)]
else:
    bkg_log10e_measured = np.asarray([])

# $$
bkg_lon_measured = []
bkg_lat_measured = []

if config_inputs['xi']!=1.0:
    
    bkg_lonlat_psf_samples =  [psf_like.sample((logeval,*coord,), 1).tolist() for logeval,coord  in notebook_tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), 
                                                                                                                  total=nbkg, ncols=80)]
    
    for bkg_lonlat_psf_sample in bkg_lonlat_psf_samples:
        bkg_lon_measured.append(bkg_lonlat_psf_sample[0])
        bkg_lat_measured.append(bkg_lonlat_psf_sample[1])

np.save(f'{datafolder}/recon_bkg_samples.npy', np.array([bkg_log10e_measured, bkg_lon_measured, bkg_lat_measured]))
np.save(f'{datafolder}/recon_sig_samples.npy', np.array([signal_log10e_measured, signal_lon_measured, signal_lat_measured]))


# $$ [markdown]
# ## Final simulation output

# $$


try:
    measured_log10e = list(signal_log10e_measured)+list(bkg_log10e_measured)
    measured_lon = list(signal_lon_measured)+list(bkg_lon_measured)
    measured_lat = list(signal_lat_measured)+list(bkg_lat_measured)
    
except:
    if type(bkg_log10e_measured)==np.float64:
        measured_log10e = list(signal_log10e_measured)
        measured_lon = list(signal_lon_measured)
        measured_lat = list(signal_lat_measured)
        measured_log10e.append(bkg_log10e_measured)
        measured_lon.append(bkg_lon_measured)
        measured_lat.append(bkg_lat_measured)
        
    elif type(signal_log10e_measured)==np.float64:
        measured_log10e = list(bkg_log10e_measured)
        measured_lon = list(bkg_lon_measured)
        measured_lat = list(bkg_lat_measured)
        measured_log10e.append(signal_log10e_measured)
        measured_lon.append(signal_lon_measured)
        measured_lat.append(signal_lat_measured)
    else:
        print('what')

# $$ [markdown]
# # <h1><b>Analysis

# $$ [markdown]
# ## Marginalisation

# $$
if nsig is None:
    nsig = len(list(measured_log10e))

logmasswindowwidth      = 10/np.sqrt(config_inputs['xi']*config_inputs['totalevents'])

logmasslowerbound       = config_inputs['logmass']-logmasswindowwidth
logmassupperbound       = config_inputs['logmass']+logmasswindowwidth

# if 1:
if logmasslowerbound<log10eaxis[0]:
    logmasslowerbound = log10eaxis[0]
# if 1:
if logmassupperbound>2:
    logmassupperbound = 2


logmassrange            = np.linspace(logmasslowerbound, logmassupperbound, config_inputs['nbins_logmass']) 

# $$
hyperparameter_likelihood_instance = hyperparameter_likelihood(priors=(DM_prior, bkg_prior,), likelihood=single_likelihood, 
                                                               dependent_axes=(log10eaxistrue,  longitudeaxistrue, latitudeaxistrue), dependent_logjacob=logjacobtrue,
                                                               hyperparameter_axes = ((logmassrange,), (None,)), numcores=config_inputs['numcores'], likelihoodnormalisation = psfnormalisationvalues+edispnormalisationvalues)

measured_log10e = [float(measured_log10e_val) for measured_log10e_val in measured_log10e]
print(f"datafolder, {datafolder}")
hyperparameter_likelihood_instance.save_data(directory_path=datafolder)

margresults = hyperparameter_likelihood_instance.nuisance_log_marginalisation(axisvals= (measured_log10e, measured_lon, measured_lat))
hyperparameter_likelihood_instance.save_data(directory_path=datafolder)






endertimer = time.perf_counter()
runtime_seconds = endertimer-startertimer
time_diff_hours = runtime_seconds//(60*60)
time_diff_minutes = (runtime_seconds- 60*60*time_diff_hours)//60
timediff_seconds = int(round(runtime_seconds- 60*60*time_diff_hours-60*time_diff_minutes))


print(f"Time to run script:{time_diff_hours} hours, {time_diff_minutes} minutes and {timediff_seconds} seconds")

