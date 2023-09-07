from BFCalc.createspectragrids import singlechannel_diffflux, getspectrafunc, darkmatterdoubleinput, energymassinputspectralfunc
from utils3d import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from astropy import units as u
from scipy import special,stats
from scipy.integrate import simps
from matplotlib import cm
from tqdm.autonotebook import tqdm as notebook_tqdm
import os, sys
import functools
from multiprocessing import Pool, freeze_support
import multiprocessing

sys.path.append("BFCalc")

if __name__=="__main__":
    
    
    try:
        identifier = sys.argv[1]
    except:
        identifier = time.strftime("%d%m%H")

    try:
        runnum = int(sys.argv[2])
    except:
        runnum = 1

    try:
        nevents = int(sys.argv[3])
    except:
        nevents = 10

    try:
        truelogmass = float(sys.argv[4]) 
    except:
        truelogmass = 0
        
    try:
        truelambda = float(sys.argv[5])
    except:
        truelambda = 0.5

    try:
        numcores = int(float(sys.argv[6]))
    except:
        numcores = 1

    try:
        os.mkdir('data')
    except:
        print("data folder already exists")
    try:
        os.mkdir(f'data/{identifier}')
    except:
        print("Stem Folder Already Exists")
        
    try:
        os.mkdir(f'data/{identifier}/singlerundata')
    except:
        print("Single Run Data Folder Already Exists")
        
    try:
        os.mkdir(f'data/{identifier}/singlerundata/{runnum}')
    except:
        raise Exception(f"The folder data/{identifier}/singlerundata/{runnum} already exists, stopping computation so files are not accidentally overwritten.")
    
    currentdirecyory = os.getcwd()
    stemdirectory = currentdirecyory+f'/data/{identifier}'
    rundirectory = f'{stemdirectory}/singlerundata/{runnum}'

    print("\nThe stem directory for the runs is: ", stemdirectory)
    print("\nThe specific run directory for the currently running analysis: ", rundirectory, '\n')
    
    
    
    
    astrophysicalbackground = np.load("unnormalised_astrophysicalbackground.npy")
    psfnormalisationvalues = np.load("psfnormalisation.npy")
    edispnormalisationvalues = np.load("edispnormalisation.npy")
    
    
    log10emeshtrue, lonmeshtrue, latmeshtrue = np.meshgrid(log10eaxistrue, longitudeaxistrue, latitudeaxistrue, indexing='ij')
    lonmeshrecon, latmeshrecon = np.meshgrid(longitudeaxis, latitudeaxis, indexing='ij')

    logjacobtrue = makelogjacob(log10eaxistrue)
    
    
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ###############   SCRIPT PARAMETER DEFINITIONS

    nsig                = int(round(truelambda*nevents))
    nbkg                = int(round((1-truelambda)*nevents))

    truelambda          = nsig/(nbkg+nsig)


    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    
    
    unnormed_logbkgpriorvalues = np.logaddexp(np.squeeze(bkgdist(log10emeshtrue, lonmeshtrue,latmeshtrue)),np.log(astrophysicalbackground))
    logbkgpriorvalues = unnormed_logbkgpriorvalues - special.logsumexp(unnormed_logbkgpriorvalues.T+logjacobtrue)
    
    
    signalspecfunc = energymassinputspectralfunc
    signalfunc = setup_full_fake_signal_dist(truelogmass, signalspecfunc)
    logsigpriorvalues = signalfunc(log10emeshtrue, lonmeshtrue,latmeshtrue)
    logsigpriorvalues = (logsigpriorvalues - special.logsumexp(logsigpriorvalues.T+logjacobtrue))

    
    
    
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ######################  True Value Simulation
    
    
    logsigbinnedprior = (logsigpriorvalues.T+logjacobtrue).T
    flattened_logsigbinnedprior = logsigbinnedprior.flatten()


    logbkgbinnedprior = (logbkgpriorvalues.T+logjacobtrue).T
    flattened_logbkgbinnedprior = logbkgbinnedprior.flatten()
    
    
    if truelambda!=0.0:
        sigresultindices = np.unravel_index(inverse_transform_sampling(flattened_logsigbinnedprior, Nsamples=nsig),logsigbinnedprior.shape)
        siglogevals = log10eaxistrue[sigresultindices[0]]
        siglonvals = longitudeaxistrue[sigresultindices[1]]
        siglatvals = latitudeaxistrue[sigresultindices[2]]
    else:
        siglogevals = np.asarray([])
        siglonvals = np.asarray([])
        siglatvals = np.asarray([])
        
    bkgresultindices = np.unravel_index(inverse_transform_sampling(flattened_logbkgbinnedprior, Nsamples=nbkg),logbkgbinnedprior.shape)
    bkglogevals = log10eaxistrue[bkgresultindices[0]]
    bkglonvals = longitudeaxistrue[bkgresultindices[1]]
    bkglatvals = latitudeaxistrue[bkgresultindices[2]]

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ######################  Reconstructed Value Simulation
    
    
    
    if truelambda!=0.0:
        signal_log10e_measured = log10eaxis[np.squeeze([inverse_transform_sampling(edisp(log10eaxis, logeval, coord)+logjacob, Nsamples=1) for logeval,coord  in notebook_tqdm(zip(siglogevals, np.array([siglonvals, siglatvals]).T), total=nsig)])]
        
        signal_spatial_indices = np.squeeze([inverse_transform_sampling(psf(np.array([lonmeshrecon.flatten(), latmeshrecon.flatten()]), logeval, coord).flatten(), Nsamples=1) for logeval, coord in notebook_tqdm(zip(siglogevals, np.array([siglonvals, siglatvals]).T), total=nsig)])
        signal_reshaped_indices = np.unravel_index(signal_spatial_indices, shape=lonmeshrecon.shape)
        signal_lon_measured = longitudeaxis[signal_reshaped_indices[0]]
        signal_lat_measured = latitudeaxis[signal_reshaped_indices[1]]
        
    else:
        signal_log10e_measured = np.asarray([])
        signal_lon_measured = np.asarray([])
        signal_lat_measured = np.asarray([])
        
    bkg_log10e_measured = log10eaxis[np.squeeze([inverse_transform_sampling(edisp(log10eaxis, logeval, coord)+logjacob, Nsamples=1) for logeval,coord  in notebook_tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), total=nbkg)])]
    
    
    bkg_spatial_indices = np.squeeze([inverse_transform_sampling(psf(np.array([lonmeshrecon.flatten(), latmeshrecon.flatten()]), logeval, coord).flatten(), Nsamples=1) for logeval, coord in notebook_tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), total=nbkg)])
    bkg_reshaped_indices = np.unravel_index(bkg_spatial_indices, shape=lonmeshrecon.shape)
    bkg_lon_measured = longitudeaxis[bkg_reshaped_indices[0]]
    bkg_lat_measured = latitudeaxis[bkg_reshaped_indices[1]]
    
    
    
    
    
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
            
    print("Done simulation.")

    
    ################################################################################################################################################
    ################################################################################################################################################
    ################################################################################################################################################
    ######## MARGINALISATION SETUP
    
    
    sigdistsetup = setup_full_fake_signal_dist
    # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
    np.seterr(divide='ignore', invalid='ignore')





    nbinslogmass            = 51
    logmasswindowwidth      = 7/np.sqrt(nsig)


    logmasslowerbound       = truelogmass-logmasswindowwidth
    logmassupperbound       = truelogmass+logmasswindowwidth



    if logmasslowerbound<log10eaxis[0]:
        logmasslowerbound = log10eaxis[0]
    if logmassupperbound>2:
        logmassupperbound = 2


    logmassrange            = np.linspace(logmasslowerbound, logmassupperbound, nbinslogmass) 







    nbinslambda            = 101
    lambdawindowwidth      = 9/np.sqrt(nevents)


    lambdalowerbound       = truelambda-lambdawindowwidth
    lambdaupperbound       = truelambda+lambdawindowwidth



    if lambdalowerbound<0:
        lambdalowerbound = 0
    if lambdaupperbound>1:
        lambdaupperbound = 1


    lambdarange            = np.linspace(lambdalowerbound, lambdaupperbound, nbinslambda) 





    ################################################################################################################################################
    ################################################################################################################################################
    ################################################################################################################################################
    ######## MARGINALISATION
    
    
    nuisance_loge_setup_mesh, nuisance_longitude_setup_mesh, nuisance_latitude_setup_mesh = np.meshgrid(log10eaxistrue, longitudeaxistrue, latitudeaxistrue, indexing='ij')

    signal_prior_matrices = []
    for logmass in logmassrange:
        priorvals= setup_full_fake_signal_dist(logmass, specfunc=signalspecfunc)(nuisance_loge_setup_mesh, nuisance_longitude_setup_mesh, nuisance_latitude_setup_mesh)
        
        priorvals = priorvals - special.logsumexp(logjacobtrue+priorvals.T)
        signal_prior_matrices.append(priorvals.T)
    
    
    marg_partial = functools.partial(diff_irf_marg, signal_prior_matrices=signal_prior_matrices, logbkgpriorvalues=logbkgpriorvalues, logmassrange=logmassrange, edispnormvalues=edispnormalisationvalues, psfnormvalues=psfnormalisationvalues)


    startertimer = time.perf_counter()

    with Pool(numcores) as pool:
            margresults = pool.map(marg_partial, tqdm(zip(measured_log10e, measured_lon, measured_lat), total=len(list(measured_log10e))))

    endertimer = time.perf_counter()
    timediffseconds = endertimer-startertimer
    
    hours = timediffseconds//(60*60)
    minutes = (timediffseconds-(60*60)*hours)//60
    seconds = timediffseconds-(60*60)*hours-60*minutes

    print(f"Marginalisation took {hours} hours, {minutes} minutes, {seconds} seconds")
    
    
    
    margresults = np.array(margresults)
    sigmargresults = np.squeeze(np.vstack(margresults[:,0])).T
    bkgmargresults = np.squeeze(np.vstack(margresults[:,1]))
    
    
    
    np.save(f'{rundirectory}/logmassrange_direct.npy',logmassrange)
    np.save(f"{rundirectory}/log_signal_marginalisations.npy", sigmargresults)
    np.save(f"{rundirectory}/log_background_marginalisations.npy", bkgmargresults)
    
    np.save(f"{rundirectory}/params.npy",         np.array([['lambda', 'nevents', 'logmass'],
                                        [truelambda, nevents, truelogmass]]))
    

    np.save(f"{rundirectory}/truesigsamples.npy", np.array([siglogevals, siglonvals, siglatvals]))
    np.save(f"{rundirectory}/meassigsamples.npy", np.array([signal_log10e_measured, signal_lon_measured, signal_lat_measured]))
    np.save(f"{rundirectory}/truebkgsamples.npy", np.array([bkglogevals, bkglonvals, bkglatvals]))
    np.save(f"{rundirectory}/measbkgsamples.npy", np.array([bkg_log10e_measured, bkg_lon_measured, bkg_lat_measured]))
    np.save(f"{rundirectory}/logbackgroundprior.npy",logbkgpriorvalues)

    
    
