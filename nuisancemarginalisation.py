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
        raise Exception("You have not input a valid identifier. Please review the name of the stemfolder for your runs.")
    try:
        runnum = sys.argv[2]
    except:
        raise Exception("Run number is not specified")
    
    try:
        totalnumberofruns = int(sys.argv[3])
    except:
        totalnumberofruns = 1
    
    try:
        nbinslogmass = int(sys.argv[4])
    except:
        nbinslogmass = 51
    
    try:
        numcores = int(sys.argv[5])
    except:
        numcores = 1

    np.seterr(divide = 'ignore')

    
    # Setting up axis values so it doesn't re-run in any parallelisation setups
    signalspecfunc = energymassinputspectralfunc

    lontrue_mesh_nuisance, logetrue_mesh_nuisance, lattrue_mesh_nuisance = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue)

    
    
    currentdirecyory = os.getcwd()
    stemdirectory = currentdirecyory+f'/data/{identifier}'
    rundirectory = f'{stemdirectory}/{runnum}'

    print("\nThe stem directory for the runs is: ", stemdirectory)
    print("\nThe specific run directory for the currently running analysis: ", rundirectory, '\n')
    
    
    siglogevals, siglonvals, siglatvals                                 = np.load(f"data/{identifier}/{runnum}/truesigsamples.npy")
    signal_log10e_measured, signal_lon_measured, signal_lat_measured    = np.load(f"data/{identifier}/{runnum}/meassigsamples.npy")
    bkglogevals, bkglonvals, bkglatvals                                 = np.load(f"data/{identifier}/{runnum}/truebkgsamples.npy")
    bkg_log10e_measured, bkg_lon_measured, bkg_lat_measured             = np.load(f"data/{identifier}/{runnum}/measbkgsamples.npy")
    truelambda, Nsamples, truelogmass                                   = np.load(f"data/{identifier}/{runnum}/params.npy")[1]
    logbkgpriorvalues                                                   = np.load(f"data/{identifier}/{runnum}/logbackgroundprior.npy")


    truelambda  = float(truelambda)
    Nsamples    = int(Nsamples)
    truelogmass = float(truelogmass)
    
    
    try:
        measured_log10e = list(signal_log10e_measured)+list(bkg_log10e_measured)
        measured_lon = list(signal_lon_measured)+list(bkg_lon_measured)
        measured_lat = list(signal_lat_measured)+list(bkg_lat_measured)
    except:
        measured_log10e = [float(signal_log10e_measured)]+list(bkg_log10e_measured)
        measured_lon = [float(signal_lon_measured)]+list(bkg_lon_measured)
        measured_lat = [float(signal_lat_measured)]+list(bkg_lat_measured)
    
    # Loading in the matrices that contain the values of the irf values for the discrete values within these scripts
        # If you cannot find these it is because you have not run the setup.py file 
    psfmatrix = np.load("psfmatrix.npy")
    edispmatrix = np.load("edispmatrix.npy")


    # Double checking the normalisation
    psfnormalisation  = special.logsumexp(psfmatrix, axis=(-2,-1))
    edispnormalisation  = special.logsumexp(edispmatrix+logjacob, axis=-1)

    edispnormalisation[edispnormalisation==-np.inf] = 0
    psfnormalisation[psfnormalisation==-np.inf] = 0   

    edispmatrix = edispmatrix-edispnormalisation[:,:,:,np.newaxis]
    psfmatrix = psfmatrix-psfnormalisation[:,:,:,np.newaxis, np.newaxis]


    psfnormalisation  = special.logsumexp(psfmatrix, axis=(-2,-1))
    edispnormalisation  = special.logsumexp(edispmatrix+logjacob, axis=-1)


    edispnormalisation[edispnormalisation==-np.inf] = 0
    psfnormalisation[psfnormalisation==-np.inf] = 0   

    edispmatrix = edispmatrix-edispnormalisation[:,:,:,np.newaxis]
    psfmatrix = psfmatrix-psfnormalisation[:,:,:,np.newaxis, np.newaxis]


    lontrue_mesh_nuisance, logetrue_mesh_nuisance, lattrue_mesh_nuisance = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue)

    irfindexlist = []

        
    with Pool(numcores) as pool: 
        
            
        for result in tqdm(pool.imap(calcrirfindices, zip(measured_log10e, np.array([measured_lon, measured_lat]).T)), total=len(list(measured_log10e)), ncols=100, desc="Calculating irf values..."):
                irfindexlist.append(result)

        pool.close() 
        
    irfindexlist = np.array(irfindexlist)

    
    bkgpriorarray  = logbkgpriorvalues
    bkgpriorarray = bkgpriorarray - special.logsumexp(bkgpriorarray.T+logjacobtrue)
    
    tempbkgmargfunc = functools.partial(marginalisenuisance, prior=bkgpriorarray, edispmatrix=edispmatrix, psfmatrix=psfmatrix)
    
    # bkgmargvals = []
    with Pool(numcores) as pool: 
        
            
        bkgmargvals = pool.map(tempbkgmargfunc, tqdm(irfindexlist, total=irfindexlist.shape[0], 
                           ncols=100, 
                           desc="Marginalising the events with the background prior."))

        pool.close() 
    bkgmargvals = np.array(bkgmargvals)
    print(f"Shape of the array containing the background marginalised probabilities: {bkgmargvals.shape}")
    
    
    
    
    sigdistsetup = setup_full_fake_signal_dist
    # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
    np.seterr(divide='ignore', invalid='ignore')




    nsig                = int(round(truelambda*(Nsamples*totalnumberofruns)))
    logmasswindowwidth      = 16/np.sqrt(nsig)

    logmasslowerbound       = truelogmass-logmasswindowwidth
    logmassupperbound       = truelogmass+logmasswindowwidth

    if logmasslowerbound<log10eaxis[0]:
        logmasslowerbound = log10eaxis[0]
    if logmassupperbound>2:
        logmassupperbound = 2

    logmassrange            = np.linspace(logmasslowerbound, logmassupperbound, nbinslogmass) 



    sigmarg_partial = functools.partial(sigmarg, specfunc=signalspecfunc, irfindexlist=irfindexlist, edispmatrix=edispmatrix, psfmatrix=psfmatrix,
                                            logetrue_mesh_nuisance=logetrue_mesh_nuisance, lontrue_mesh_nuisance=lontrue_mesh_nuisance, 
                                            lattrue_mesh_nuisance=lattrue_mesh_nuisance, logjacobtrue=logjacobtrue)
    print(time.strftime("The time before signal marginalisation is %d of %b, at %H:%M:%S"))
    sigmargresults = []
    with Pool(numcores) as pool:
        for result in pool.imap(sigmarg_partial, tqdm(logmassrange, ncols=100, total=logmassrange.shape[0])):
            sigmargresults.append(result)
            
    print(time.strftime("The time after signal marginalisation is %d of %b, at %H:%M:%S"))
    print("Finished signal marginalisation. \nNow converting to numpy arrays and saving the result...")
    
    signal_log_marginalisationvalues = np.array(sigmargresults)
    
    print(f"Shape of the array containing the signal marginalised probabilities: {signal_log_marginalisationvalues.shape}")


    



        
    np.save(f'{rundirectory}/logmassrange_direct.npy',logmassrange)
    np.save(f"{rundirectory}/irfindexlist.npy", irfindexlist)
    np.save(f"{rundirectory}/log_signal_marginalisations.npy", signal_log_marginalisationvalues)
    np.save(f"{rundirectory}/log_background_marginalisations.npy", bkgmargvals)