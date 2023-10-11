#   [markdown]
# # <h1><b><I> General Setup

#   [markdown]
# ## Imports and general setup

#  
import os, sys
sys.path.append("gammabayes/BFCalc/")
sys.path.append("gammabayes")

from gammabayes.BFCalc.createspectragrids import singlechannel_diffflux, getspectrafunc, darkmatterdoubleinput, energymassinputspectralfunc
from gammabayes.utils.utils import log10eaxistrue, longitudeaxistrue, latitudeaxistrue, log10eaxis, longitudeaxis, latitudeaxis, time,psf, edisp, bkgdist, interpolate, special, integrate
from gammabayes.utils.utils import SkyCoord, WcsGeom, inverse_transform_sampling, tqdm
from gammabayes.load_package_data import edispnormalisationvalues, psfnormalisationvalues, astrophysicalbackground
from gammabayes.hyperparameter_likelihood import hyperparameter_likelihood
from gammabayes.prior import discrete_logprior
from gammabayes.likelihood import discrete_loglikelihood
from gammabayes.utils.utils import edisp_test, psf_test, log10eaxis, longitudeaxis, latitudeaxis
from gammabayes.utils.utils import psf_efficient, edisp_efficient, edisp_test, psf_test, single_likelihood
from gammabayes.utils.utils import read_config_file, check_necessary_config_inputs

from gammabayes.SS_DM_Prior import SS_DM_dist
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from astropy import units as u
from scipy import special,stats
from scipy.integrate import simps
from tqdm import tqdm

import functools
from multiprocessing import Pool, freeze_support
import multiprocessing
import pandas as pd
import yaml


#  

print(psfnormalisationvalues.shape)

#  
log10emeshtrue, lonmeshtrue, latmeshtrue = np.meshgrid(log10eaxistrue, longitudeaxistrue, latitudeaxistrue, indexing='ij')
lonmeshrecon, latmeshrecon = np.meshgrid(longitudeaxis, latitudeaxis, indexing='ij')

logjacobtrue = np.meshgrid(np.log(10**log10eaxistrue), longitudeaxistrue, latitudeaxistrue, indexing='ij')[0]


print(lonmeshtrue.shape, lonmeshrecon.shape)

# ## Script Parameter Setup

#  


inputs = read_config_file(sys.argv[1])

check_necessary_config_inputs(inputs)

print(inputs)

nsig                = int(round( inputs['xi'] *inputs['Nevents']))
nbkg                = int(round((1- inputs['xi'])*inputs['Nevents']))

true_xi          = nsig/(nbkg+nsig)


try:
    os.mkdir('data')
except:
    print('data folder exists')

try:
    os.mkdir(f"data/{inputs['identifier']}")
except:
    print('stem folder exists')

stemdatafoldername = f"data/{inputs['identifier']}/singlerundata"

try:
    os.mkdir(f'{stemdatafoldername}')
except:
    print('stem data folder exists')

datafolder = f"{stemdatafoldername}/{inputs['runnumber']}"

try:
    os.mkdir(datafolder)
except:
    print('Single run data folder exists')

if os.path.exists(f"{datafolder}/margresultsarray.npy"):
    raise Exception(f"margresultsarray.npy (final result) already exists. Exiting to not overwrite data.")
    

# To make the config file accessible from the stem data folder
with open(stemdatafoldername+'/inputconfig.yaml', 'w') as file:
    yaml.dump(inputs, file, default_flow_style=False)
    


startertimer = time.perf_counter()
print(startertimer)

# # # # # # # # # # #
# ## Setup
# # # # # # # # # # #

# ### Background setup

#  
unnormed_logbkgpriorvalues = np.logaddexp(np.squeeze(bkgdist(log10emeshtrue, lonmeshtrue,latmeshtrue)),np.log(astrophysicalbackground))


logbkgpriorvalues = unnormed_logbkgpriorvalues - special.logsumexp(unnormed_logbkgpriorvalues+logjacobtrue)

logbkgpriorvalues.shape


nuisancemesh = np.meshgrid(log10eaxistrue, longitudeaxistrue, latitudeaxistrue, indexing='ij')

unnormed_logbkgpriorvalues = np.logaddexp(np.squeeze(bkgdist(*nuisancemesh)),np.log(astrophysicalbackground))


logbkgfunc_annoying = interpolate.RegularGridInterpolator((log10eaxistrue, longitudeaxistrue, latitudeaxistrue), 
                                                          np.exp(unnormed_logbkgpriorvalues))
logbkgfunc = lambda logenergy, longitude, latitude: np.log(logbkgfunc_annoying((logenergy, longitude, latitude)))


bkg_prior = discrete_logprior(logfunction=logbkgfunc, name='Background Prior',
                               axes=(log10eaxistrue, longitudeaxistrue, latitudeaxistrue,), 
                               axes_names=['energy', 'lon', 'lat'], logjacob=logjacobtrue)

# ### Signal Setup

#  
from gammapy.astro.darkmatter import profiles
density_profile_list = {'einasto':profiles.EinastoProfile(), 
                        'nfw':profiles.NFWProfile(),
                        'isothermal':profiles.IsothermalProfile(),
                        'burkert':profiles.BurkertProfile(),
                        'moore':profiles.MooreProfile()}



SS_DM_dist_instance= SS_DM_dist(longitudeaxistrue, latitudeaxistrue, density_profile_list[inputs['dmdensity_profile'].lower()])
logDMpriorfunc = SS_DM_dist_instance.func_setup()

#  
DM_prior = discrete_logprior(logfunction=logDMpriorfunc, name='Scalar Singlet Dark Matter Prior',
                               axes=(log10eaxistrue, longitudeaxistrue, latitudeaxistrue,), axes_names=['energy', 'lon', 'lat'],
                               default_hyperparameter_values=(inputs['logmass'],), hyperparameter_names=['mass'], logjacob=logjacobtrue)


# # # # # # # # # # # # # # 
# # Simulation 
# # # # # # # # # # # # # # 


# ## True Value Simulation

#  
if  inputs['xi']!=0.0:
    siglogevals,siglonvals,siglatvals  = DM_prior.sample(nsig)
else:
    siglogevals = np.asarray([])
    siglonvals = np.asarray([])
    siglatvals = np.asarray([])



#  

if  inputs['xi']!=1.0:
    bkglogevals,bkglonvals,bkglatvals  = bkg_prior.sample(nbkg)
else:
    bkglogevals = np.asarray([])
    bkglonvals = np.asarray([])
    bkglatvals = np.asarray([])


np.save(f'{datafolder}/true_bkg_samples.npy', np.array([bkglogevals, bkglonvals, bkglatvals]))
np.save(f'{datafolder}/true_sig_samples.npy', np.array([siglogevals, siglonvals, siglatvals]))



# ## Reconstructed Value Simulation
#  
logjacob = np.meshgrid(np.log(10**log10eaxis), longitudeaxis, latitudeaxis, indexing='ij')[0]

#  
logjacob = np.log(10**log10eaxis)
edisp_like = discrete_loglikelihood(logfunction=edisp_test, 
                                    axes=(log10eaxis,), axes_names='log10E recon',
                                    name='energy dispersion',
                                    dependent_axes=(log10eaxistrue, longitudeaxistrue, latitudeaxistrue,), logjacob=logjacob,
                                    dependent_axes_names = ['log10E true', 'lon', 'lat'])

#  
psf_like = discrete_loglikelihood(logfunction=psf_test, 
                                    axes=(longitudeaxis, latitudeaxis), axes_names=['longitude recon', 'latitude recon'],
                                    name='point spread function ',
                                    dependent_axes=(log10eaxistrue, longitudeaxistrue, latitudeaxistrue,),
                                    dependent_axes_names = ['log10E true', 'lon', 'lat'])
psf_like

# ### Signal

#  

if  inputs['xi']!=0.0:
    # sig_log10e_edisp_samples = [edisp_like.sample(signal_event_tuple, 1).tolist() for signal_event_tuple in tqdm(sig_samples.T)]
    signal_log10e_measured = [np.squeeze(edisp_like.sample((logeval,*coord,), numsamples=1)) for logeval,coord  in  tqdm(zip(siglogevals, np.array([siglonvals, 
                                                                                                                                                    siglatvals]).T), 
                                                                                                                         total=nsig, 
                                                                                                                         ncols=80)]
else:
    signal_log10e_measured = np.asarray([])

#  

signal_lon_measured = []
signal_lat_measured = []

if  inputs['xi']!=0:
    
    sig_lonlat_psf_samples =  [psf_like.sample((logeval,*coord,), 1).tolist() for logeval,coord  in  tqdm(zip(siglogevals, np.array([siglonvals, 
                                                                                                                                     siglatvals]).T), 
                                                                                                          total=nsig, 
                                                                                                          ncols=80)]
    
    for sig_lonlat_psf_sample in sig_lonlat_psf_samples:
        signal_lon_measured.append(sig_lonlat_psf_sample[0])
        signal_lat_measured.append(sig_lonlat_psf_sample[1])

# ### Background

#  
if  inputs['xi']!=1.0:
    # sig_log10e_edisp_samples = [edisp_like.sample(signal_event_tuple, 1).tolist() for signal_event_tuple in tqdm(sig_samples.T)]
    bkg_log10e_measured = [np.squeeze(edisp_like.sample((logeval,*coord,), numsamples=1)) for logeval,coord  in  tqdm(zip(bkglogevals, np.array([bkglonvals, 
                                                                                                                                                 bkglatvals]).T), 
                                                                                                                      total=nbkg,
                                                                                                                      ncols=80)]
else:
    bkg_log10e_measured = np.asarray([])

#  
bkg_lon_measured = []
bkg_lat_measured = []

if  inputs['xi']!=1.0:
    
    bkg_lonlat_psf_samples =  [psf_like.sample((logeval,*coord,), 1).tolist() for logeval,coord  in  tqdm(zip(bkglogevals, np.array([bkglonvals, 
                                                                                                                                     bkglatvals]).T), 
                                                                                                          total=nbkg, 
                                                                                                          ncols=80)]
    
    for bkg_lonlat_psf_sample in bkg_lonlat_psf_samples:
        bkg_lon_measured.append(bkg_lonlat_psf_sample[0])
        bkg_lat_measured.append(bkg_lonlat_psf_sample[1])


np.save(f'{datafolder}/recon_bkg_samples.npy', np.array([bkg_log10e_measured, bkg_lon_measured, bkg_lat_measured]))
np.save(f'{datafolder}/recon_sig_samples.npy', np.array([signal_log10e_measured, signal_lon_measured, signal_lat_measured]))


# ## Final simulation output

#  


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
# # # # # # # # # # # # # # # # # #
# # # # # # # # Analysis
# # # # # # # # # # # # # # # # # #
# ## Marginalisation

#  

if nsig is None:
    nsig = len(list(measured_log10e))

if inputs['xi']>1e-2:
    logmasswindowwidth      = 10/np.sqrt(inputs['xi']*inputs['totalevents'])

    logmasslowerbound       = inputs['logmass']-logmasswindowwidth
    logmassupperbound       = inputs['logmass']+logmasswindowwidth
else:
    logmasslowerbound = log10eaxis[0]
    logmassupperbound = 2


if logmasslowerbound<log10eaxis[0]:
    logmasslowerbound = log10eaxis[0]
if logmassupperbound>2:
    logmassupperbound = 2


logmassrange            = np.linspace(logmasslowerbound, logmassupperbound, inputs['nbins_logmass']) 

#  
hyperparameter_likelihood_instance = hyperparameter_likelihood(priors=(DM_prior, bkg_prior,), likelihood=single_likelihood, 
                                                               dependent_axes=(log10eaxistrue,  longitudeaxistrue, latitudeaxistrue), 
                                                               dependent_logjacob=logjacobtrue,
                                                               hyperparameter_axes = (logmassrange, (None,)), 
                                                               numcores=inputs['numcores'], 
                                                               likelihoodnormalisation = psfnormalisationvalues+edispnormalisationvalues)

measured_log10e = [float(measured_log10e_val) for measured_log10e_val in measured_log10e]
margresults = hyperparameter_likelihood_instance.full_obs_marginalisation(axisvals= (measured_log10e, measured_lon, measured_lat))
margresultsarray = np.array(margresults)

#  

np.save(f'{datafolder}/logmassrange.npy', logmassrange)
np.save(f'{datafolder}/margresultsarray.npy', margresultsarray)


endertimer = time.perf_counter()
print(endertimer-startertimer)