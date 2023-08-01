from BFCalc.createspectragrids import energymassinputspectralfunc
from utils3d import *
import numpy as np
from astropy import units as u
from scipy import special,stats
from tqdm import tqdm
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
        print("data folder already exists (good)")
    try:
        os.mkdir(f'data/{identifier}')
    except:
        print("Stem Folder Already Exists (good)")
    try:
        os.mkdir(f'data/{identifier}/{runnum}')
    except:
        raise Exception(f"The folder data/{identifier}/{runnum} already exists, stopping computation so files are not accidentally overwritten.")

    astrophysicalbackground = np.load("unnormalised_astrophysicalbackground.npy")
    
    
    
    lonmeshtrue, log10emeshtrue, latmeshtrue = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue)
    latmeshrecon, lonmeshrecon = np.meshgrid(latitudeaxis, longitudeaxis)

    logjacobtrue = makelogjacob(log10eaxistrue)  

    print(lonmeshtrue.shape, lonmeshrecon.shape)  
    
    
    
    Nsamples=nevents
    nsig                = int(round(truelambda*Nsamples))
    nbkg                = int(round((1-truelambda)*Nsamples))

    # Accounting for any rounding errors
    truelambda          = nsig/(nbkg+nsig)
    numcores            = 2
    
    
    #########################################################################################################
    ### Background Setup
    logbkgpriorvalues = np.squeeze(bkgdist(log10emeshtrue, lonmeshtrue,latmeshtrue))+np.log(astrophysicalbackground)


    logbkgpriorvalues = logbkgpriorvalues - special.logsumexp(logbkgpriorvalues.T+logjacobtrue)

    print(f"Shape of the background prior: {logbkgpriorvalues.shape}")
    
    
    
    
    
    #########################################################################################################
    ### Signal Setup
    
    
    signalspecfunc = energymassinputspectralfunc
    signalfunc = setup_full_fake_signal_dist(truelogmass, signalspecfunc)
    
    
    logsigpriorvalues = signalfunc(log10emeshtrue, lonmeshtrue,latmeshtrue)
    logsigpriorvalues = (logsigpriorvalues - special.logsumexp(logsigpriorvalues.T+logjacobtrue))

    
    print(f"Shape of the signal prior: {logsigpriorvalues.shape}")


    #########################################################################################################
    ### Flattening the priors and applying the jacobian for use in the inverse transform sampling
    logsigbinnedprior = (logsigpriorvalues.T+logjacobtrue).T
    flattened_logsigbinnedprior = logsigbinnedprior.flatten()


    logbkgbinnedprior = (logbkgpriorvalues.T+logjacobtrue).T
    flattened_logbkgbinnedprior = logbkgbinnedprior.flatten()
    
    #########################################################################################################
    ### True Event Simulation
    print("\n\nTrue Event Simulation\n\n")
    
    # Simulating the signal events
    print("\n\nSIGNAL\n\n")
    sigresultindices = np.unravel_index(inverse_transform_sampling(flattened_logsigbinnedprior, Nsamples=nsig),logsigbinnedprior.shape)
    siglogevals = log10eaxistrue[sigresultindices[0]]
    siglonvals = longitudeaxistrue[sigresultindices[1]]
    siglatvals = latitudeaxistrue[sigresultindices[2]]
    
    
    # Simulating the background events
    print("\n\nBACKGROUND\n\n")
    bkgresultindices = np.unravel_index(inverse_transform_sampling(flattened_logbkgbinnedprior, Nsamples=nbkg),logbkgbinnedprior.shape)
    bkglogevals = log10eaxistrue[bkgresultindices[0]]
    bkglonvals = longitudeaxistrue[bkgresultindices[1]]
    bkglatvals = latitudeaxistrue[bkgresultindices[2]]
    
    #########################################################################################################
    ### Pseudo-Measured Event Simulation
    print("\n\nPseudo-Measured Event Simulation\n\n")

    ### Simulating measured signal events
    
    # Simulating the measured energy signal events with noise added via the energy dispersion
    print("\n\nSIGNAL ENERGY\n\n")

    signal_log10e_measured = log10eaxis[np.squeeze([inverse_transform_sampling(edisp(log10eaxis, logeval, coord)+logjacob, Nsamples=1) for logeval,coord  in tqdm(zip(siglogevals, np.array([siglonvals, siglatvals]).T), total=nsig)])]
    
    
    # Simulating the measured spatial signal events with noise added via the point spread function
    print("\n\nSIGNAL SPATIAL\n\n")

    signal_spatial_indices = np.squeeze([inverse_transform_sampling(psf(np.array([lonmeshrecon.flatten(), latmeshrecon.flatten()]), coord, logeval).flatten(), Nsamples=1) for logeval, coord in tqdm(zip(siglogevals, np.array([siglonvals, siglatvals]).T), total=nsig)])
    signal_reshaped_indices = np.unravel_index(signal_spatial_indices, shape=lonmeshrecon.shape)
    signal_lon_measured = longitudeaxis[signal_reshaped_indices[0]]
    signal_lat_measured = latitudeaxis[signal_reshaped_indices[1]]
    
    
    ### Simulating measured backgroudn events
    
    # Simulating the measured energy background events with noise added via the energy dispersion
    print("\n\BACKGROUND ENERGY\n\n")

    bkg_log10e_measured = log10eaxis[np.squeeze([inverse_transform_sampling(edisp(log10eaxis, logeval, coord)+logjacob, Nsamples=1) for logeval,coord  in tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), total=nbkg)])]


    # Simulating the measured spatial background events with noise added via the point spread function
    print("\n\BACKRGOUND SPATIAL\n\n")
    bkg_spatial_indices = np.squeeze([inverse_transform_sampling(psf(np.array([lonmeshrecon.flatten(), latmeshrecon.flatten()]), coord, logeval).flatten(), Nsamples=1) for logeval, coord in tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), total=nbkg)])
    bkg_reshaped_indices = np.unravel_index(bkg_spatial_indices, shape=lonmeshrecon.shape)
    bkg_lon_measured = longitudeaxis[bkg_reshaped_indices[0]]
    bkg_lat_measured = latitudeaxis[bkg_reshaped_indices[1]]

    
    
    print("\n\nSaving Results\n\n")


    np.save(f"data/{identifier}/{runnum}/truesigsamples.npy", np.array([siglogevals, siglonvals, siglatvals]))
    np.save(f"data/{identifier}/{runnum}/meassigsamples.npy", np.array([signal_log10e_measured, signal_lon_measured, signal_lat_measured]))
    np.save(f"data/{identifier}/{runnum}/truebkgsamples.npy", np.array([bkglogevals, bkglonvals, bkglatvals]))
    np.save(f"data/{identifier}/{runnum}/measbkgsamples.npy", np.array([bkg_log10e_measured, bkg_lon_measured, bkg_lat_measured]))
    np.save(f"data/{identifier}/{runnum}/params.npy",         np.array([['lambda', 'Nsamples', 'logmass'],
                                            [truelambda, nevents, truelogmass]]))
    np.save(f"data/{identifier}/{runnum}/logbackgroundprior.npy",logbkgpriorvalues)

    print("Done simulation.")