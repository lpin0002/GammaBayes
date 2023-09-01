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



    nuisance_loge_setup_mesh, nuisance_longitude_setup_mesh, nuisance_latitude_setup_mesh = np.meshgrid(log10eaxistrue, longitudeaxistrue, latitudeaxistrue, indexing='ij')

    signal_prior_matrices = []
    for logmass in logmassrange:
        priorvals= setup_full_fake_signal_dist(logmass, specfunc=signalspecfunc)(nuisance_loge_setup_mesh, nuisance_longitude_setup_mesh, nuisance_latitude_setup_mesh)
        
        priorvals = priorvals - special.logsumexp(logjacobtrue+priorvals.T)
        signal_prior_matrices.append(priorvals.T)
    
    
    marg_partial = functools.partial(efficient_marg, signal_prior_matrices=signal_prior_matrices, logbkgpriorvalues=logbkgpriorvalues, logmassrange=logmassrange, edispmatrix=edispmatrix, psfmatrix=psfmatrix)


    startertimer = time.perf_counter()

    with Pool(numcores) as pool:
            margresults = pool.map(marg_partial, tqdm(zip(measured_log10e, measured_lon, measured_lat), total=len(list(measured_log10e))))
            
            
            
    margresults = np.array(margresults)
    sigmargresults = np.squeeze(np.vstack(margresults[:,0])).T
    bkgmargresults = np.squeeze(np.vstack(margresults[:,1]))

    endertimer = time.perf_counter()
    timediffseconds = endertimer-startertimer
    
    hours = timediffseconds//(60*60)
    minutes = (timediffseconds-(60*60)*hours)//60
    seconds = timediffseconds-(60*60)*hours-60*minutes

    print(f"Marginalisation took {hours} hours, {minutes} minutes, {seconds} seconds")


    



        
    np.save(f'{rundirectory}/logmassrange_direct.npy',logmassrange)
    np.save(f"{rundirectory}/log_signal_marginalisations.npy", sigmargresults)
    np.save(f"{rundirectory}/log_background_marginalisations.npy", bkgmargresults)