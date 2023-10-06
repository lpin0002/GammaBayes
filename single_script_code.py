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
from gammabayes.load_package_data import edispnormalisationvalues, psfnormalisationvalues, astrophysicalbackground, log_single_likelihood
from gammabayes.hyperparameter_likelihood import hyperparameter_likelihood
from gammabayes.prior import discrete_logprior
from gammabayes.likelihood import discrete_loglikelihood
from gammabayes.utils.utils import edisp_test, psf_test, log10eaxis, longitudeaxis, latitudeaxis, single_likelihood
from gammabayes.utils.utils import psf_efficient, edisp_efficient, edisp_test, psf_test
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

if __name__=="__main__":
    #  
    print("\n\n")

    #  
    log10emeshtrue, lonmeshtrue, latmeshtrue = np.meshgrid(log10eaxistrue, longitudeaxistrue, latitudeaxistrue, indexing='ij')
    lonmeshrecon, latmeshrecon = np.meshgrid(longitudeaxis, latitudeaxis, indexing='ij')

    logjacobtrue = np.meshgrid(np.log(10**log10eaxistrue), longitudeaxistrue, latitudeaxistrue, indexing='ij')[0]

 
    # ## Script Parameter Setup

    #  


    inputs = read_config_file(sys.argv[1])

    check_necessary_config_inputs(inputs)


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
    print('\n\n')

##########################################################################################
    print("Initiating background prior class instance")
    bkg_prior = discrete_logprior(logfunction=logbkgfunc, name='Background Prior',
                                axes=(log10eaxistrue, longitudeaxistrue, latitudeaxistrue,), 
                                axes_names=['energy', 'lon', 'lat'], logjacob=logjacobtrue)
    print('\n\n')
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

##########################################################################################
    print("Initiating signal prior class instance")
    DM_prior = discrete_logprior(logfunction=logDMpriorfunc, name='Scalar Singlet Dark Matter Prior',
                                axes=(log10eaxistrue, longitudeaxistrue, latitudeaxistrue,), axes_names=['energy', 'lon', 'lat'],
                                default_hyperparameter_values=(inputs['logmass'],), hyperparameter_names=['mass'], logjacob=logjacobtrue)
    print('\n\n')

    # # # # # # # # # # # # # # 
    # # Simulation 
    # # # # # # # # # # # # # # 


    # ## True Value Simulation

    #  
    if  inputs['xi']!=0.0:
##########################################################################################
        siglogevals,siglonvals,siglatvals  = DM_prior.sample(nsig)
    else:
        siglogevals = np.asarray([])
        siglonvals = np.asarray([])
        siglatvals = np.asarray([])


    if  inputs['xi']!=1.0:
##########################################################################################
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

    print("Initiating energy dispersion likelihood class instance")


##########################################################################################
    edisp_like = discrete_loglikelihood(logfunction=edisp_test, 
                                        axes=(log10eaxis,), axes_names='log10E recon',
                                        name='energy dispersion',
                                        dependent_axes=(log10eaxistrue, longitudeaxistrue, latitudeaxistrue,), logjacob=logjacob,
                                        dependent_axes_names = ['log10E true', 'lon', 'lat'])
    print('\n\n')

    print("Initiating PSF likelihood class instance")

##########################################################################################
    psf_like = discrete_loglikelihood(logfunction=psf_test, 
                                        axes=(longitudeaxis, latitudeaxis), axes_names=['longitude recon', 'latitude recon'],
                                        name='point spread function ',
                                        dependent_axes=(log10eaxistrue, longitudeaxistrue, latitudeaxistrue,),
                                        dependent_axes_names = ['log10E true', 'lon', 'lat'])
    print('\n\n')
    # ### Signal

    #  
    before_measured_sampling = time.perf_counter()

    if  inputs['xi']!=0.0:
##########################################################################################
        signal_log10e_measured = [np.squeeze(edisp_like.sample((logeval,*coord,))) for logeval,coord  in  tqdm(zip(siglogevals, np.array([siglonvals, 
                                                                                                                                                        siglatvals]).T), 
                                                                                                                            total=nsig, 
                                                                                                                            ncols=80)]
    else:
        signal_log10e_measured = np.asarray([])

    #  

    signal_lon_measured = []
    signal_lat_measured = []

    if  inputs['xi']!=0:
##########################################################################################
        sig_lonlat_psf_samples =  [psf_like.sample((logeval,*coord,)) for logeval,coord  in  tqdm(zip(siglogevals, np.array([siglonvals, 
                                                                                                                                        siglatvals]).T), 
                                                                                                            total=nsig, 
                                                                                                            ncols=80)]
        
        for sig_lonlat_psf_sample in sig_lonlat_psf_samples:
            signal_lon_measured.append(sig_lonlat_psf_sample[0])
            signal_lat_measured.append(sig_lonlat_psf_sample[1])

    # ### Background

    #  
    if  inputs['xi']!=1.0:
##########################################################################################
        bkg_log10e_measured = [np.squeeze(edisp_like.sample((logeval,*coord,))) for logeval,coord  in  tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), 
                                                                                                            total=nbkg,
                                                                                                            ncols=80)]
    else:
        bkg_log10e_measured = np.asarray([])

    #  
    bkg_lon_measured = []
    bkg_lat_measured = []

    if  inputs['xi']!=1.0:
        
##########################################################################################
        bkg_lonlat_psf_samples =  [psf_like.sample((logeval,*coord,)) for logeval, coord  in  tqdm(zip(bkglogevals, np.array([bkglonvals,bkglatvals]).T), 
                                                                                                   total=nbkg,
                                                                                                   ncols=80)]
        for idx, bkg_lonlat_psf_sample in enumerate(bkg_lonlat_psf_samples):
            if idx==0:
                print(bkg_lonlat_psf_sample)
            bkg_lon_measured.append(bkg_lonlat_psf_sample[0])
            bkg_lat_measured.append(bkg_lonlat_psf_sample[1])


    after_measured_sampling = time.perf_counter()


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

##########################################################################################
    hyperparameter_likelihood_instance = hyperparameter_likelihood(priors=(DM_prior, bkg_prior,), likelihood=log_single_likelihood, 
                                                                dependent_axes=(log10eaxistrue,  longitudeaxistrue, latitudeaxistrue), dependent_logjacob=logjacobtrue,
                                                                hyperparameter_axes_tuple = ((logmassrange,), (None,)), 
                                                                numcores=inputs['numcores'], likelihoodnormalisation = psfnormalisationvalues+edispnormalisationvalues)

    measured_log10e = [float(measured_log10e_val) for measured_log10e_val in measured_log10e]

    before_nuisance_marg = time.perf_counter()
##########################################################################################
    margresults = hyperparameter_likelihood_instance.full_obs_marginalisation(axisvals= (measured_log10e, measured_lon, measured_lat))

    after_nuisance_marg = time.perf_counter()

    #  

    hyperparameter_likelihood_instance.save_data(directory_path=datafolder)


    endertimer = time.perf_counter()
    print(f"total run time: {endertimer-startertimer}")
    print(f"time to do measured value sampling: {after_measured_sampling-before_measured_sampling}")
    print(f"time to do nuisance marginalisation: {after_nuisance_marg-before_nuisance_marg}")


    nbinslambda            = 81
    xi_windowwidth      = 9/np.sqrt(inputs['totalevents'])


    xilowerbound       = inputs['xi']-xi_windowwidth
    xiupperbound       = inputs['xi']+xi_windowwidth



    if xilowerbound<0:
        xilowerbound = 0
    if xiupperbound>1:
        xiupperbound = 1


    xi_range            = np.linspace(xilowerbound, xiupperbound, inputs['nbins_xi']) 


    hyperparameter_likelihood_instance.create_mixture_log_posterior(mixture_axes = (xi_range, 1-xi_range,))
    log_posterior = hyperparameter_likelihood_instance.unnormed_log_posterior
    log_posterior = np.squeeze(log_posterior - special.logsumexp(log_posterior))
    plt.figure()
    plt.pcolormesh(log_posterior)
    plt.show()