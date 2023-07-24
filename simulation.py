from BFCalc.createspectragrids import darkmatterdoubleinput
from utils3d import *
import numpy as np
from scipy import special
from tqdm import tqdm
import os, sys, time
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
        lambdaval = float(sys.argv[5])
    except:
        lambdaval = 0.5

    # try:
    #     numcores = int(float(sys.argv[6]))
    # except:
    #     numcores = 10

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

    lonmeshtrue, log10emeshtrue, latmeshtrue = np.meshgrid(spatialaxistrue, log10eaxistrue, spatialaxistrue)
    lonmeshrecon, latmeshrecon = np.meshgrid(spatialaxis, spatialaxis)

    logjacobtrue = makelogjacob(log10eaxistrue)   
    
    
    Nsamples=nevents

    
    Nsamples=nevents
    truelambda          = lambdaval
    nsig                = int(round(truelambda*Nsamples))
    nbkg                = int(round((1-truelambda)*Nsamples))


    # Accounting for rounding errors
    truelambda          = nsig/(nbkg+nsig)
    numcores            = 9
    
    truelogmassval = truelogmass
    
    
    logbkgpriorvalues = np.squeeze(bkgdist(log10emeshtrue, lonmeshtrue,latmeshtrue))

    logbkgpriorvalues = logbkgpriorvalues - special.logsumexp(logbkgpriorvalues.T+logjacobtrue)

    print(f"Background prior array shape: {logbkgpriorvalues.shape}")
    
    
    signalspecfunc = darkmatterdoubleinput
    
    signalfunc = setup_full_fake_signal_dist(truelogmassval, signalspecfunc)
    
    
    logsigpriorvalues = signalfunc(log10emeshtrue, lonmeshtrue,latmeshtrue)
    
    logsigpriorvalues = (logsigpriorvalues - special.logsumexp(logsigpriorvalues.T+logjacobtrue))

    print(f"Signal prior array shape: {logsigpriorvalues.shape}")
    
    
    
    # Flattening the prior arrays for use within the inverse transform sampling function which only takes in 1d arrays
        # And adding the jacobian to 'simulate' the integral
    logsigbinnedprior = (logsigpriorvalues.T+logjacobtrue).T
    flattened_logsigbinnedprior = logsigbinnedprior.flatten()


    logbkgbinnedprior = (logbkgpriorvalues.T+logjacobtrue).T
    flattened_logbkgbinnedprior = logbkgbinnedprior.flatten()
    
    
    
    ##############################################################################################################################
    ##############################################################################################################################
    ### Signal event simulation    
    
    if truelambda>0:
        sigresultindices = np.unravel_index(inverse_transform_sampling(flattened_logsigbinnedprior, Nsamples=nsig),logsigbinnedprior.shape)
        siglogevals = log10eaxistrue[sigresultindices[0]]
        siglonvals = spatialaxistrue[sigresultindices[1]]
        siglatvals = spatialaxistrue[sigresultindices[2]]
        
        
        
        # Signal measured energy simulation
    
        signal_log10e_measured = log10eaxis[np.squeeze([inverse_transform_sampling(edisp(log10eaxis, logeval, coord)+logjacob, Nsamples=1) for logeval,coord  in tqdm(zip(siglogevals, np.array([siglonvals, siglatvals]).T), total=nsig)])]
        
        # Signal measured sky position simulation
        signal_spatial_indices = np.squeeze([inverse_transform_sampling(psf(np.array([lonmeshrecon.flatten(), latmeshrecon.flatten()]), coord, logeval).flatten(), Nsamples=1) for logeval, coord in tqdm(zip(siglogevals, np.array([siglonvals, siglatvals]).T), total=nsig)])
        signal_reshaped_indices = np.unravel_index(signal_spatial_indices, shape=lonmeshrecon.shape)
        signal_lon_measured = spatialaxis[signal_reshaped_indices[0]]
        signal_lat_measured = spatialaxis[signal_reshaped_indices[1]]
        
        
    else:
        siglogevals = np.array([])
        siglonvals = np.array([])
        siglatvals = np.array([])
        
        signal_log10e_measured = np.array([])
        signal_lon_measured = np.array([])
        signal_lat_measured = np.array([])
    
    ##############################################################################################################################
    ##############################################################################################################################
    ### Background event simulation
    
    
    if truelambda<1:
        bkgresultindices = np.unravel_index(inverse_transform_sampling(flattened_logbkgbinnedprior, Nsamples=nbkg),logbkgbinnedprior.shape)
        bkglogevals = log10eaxistrue[bkgresultindices[0]]
        bkglonvals = spatialaxistrue[bkgresultindices[1]]
        bkglatvals = spatialaxistrue[bkgresultindices[2]]
        
        
        # Background measured energy simulation
        bkg_log10e_measured = log10eaxis[np.squeeze([inverse_transform_sampling(edisp(log10eaxis, logeval, coord)+logjacob, Nsamples=1) for logeval,coord  in tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), total=nbkg)])]
        
        
        # Background measured sky position simulation
        bkg_spatial_indices = np.squeeze([inverse_transform_sampling(psf(np.array([lonmeshrecon.flatten(), latmeshrecon.flatten()]), coord, logeval).flatten(), Nsamples=1) for logeval, coord in tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), total=nbkg)])
        bkg_reshaped_indices = np.unravel_index(bkg_spatial_indices, shape=lonmeshrecon.shape)
        bkg_lon_measured = spatialaxis[bkg_reshaped_indices[0]]
        bkg_lat_measured = spatialaxis[bkg_reshaped_indices[1]]
        
        
        
    else:
        bkglogevals = np.array([])
        bkglonvals = np.array([])
        bkglatvals = np.array([])
        
        bkg_log10e_measured = np.array([])
        bkg_lon_measured = np.array([])
        bkg_lat_measured = np.array([])
    
    


    np.save(f"data/{identifier}/{runnum}/truesigsamples.npy", np.array([siglogevals, siglonvals, siglatvals]))
    np.save(f"data/{identifier}/{runnum}/meassigsamples.npy", np.array([signal_log10e_measured, signal_lon_measured, signal_lat_measured]))
    np.save(f"data/{identifier}/{runnum}/truebkgsamples.npy", np.array([bkglogevals, bkglonvals, bkglatvals]))
    np.save(f"data/{identifier}/{runnum}/measbkgsamples.npy", np.array([bkg_log10e_measured, bkg_lon_measured, bkg_lat_measured]))
    np.save(f"data/{identifier}/{runnum}/params.npy",         np.array([['lambda', 'Nsamples', 'logmass'],
                                            [lambdaval, nevents, truelogmass]]))
    np.save(f"data/{identifier}/{runnum}/logbackgroundprior.npy",logbkgpriorvalues)

    print("Done simulation.")