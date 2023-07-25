from utils3d import *
from BFCalc.createspectragrids import darkmatterdoubleinput
import numpy as np
from scipy import special
from tqdm import tqdm
import os, sys, time, functools
from multiprocessing import Pool, freeze_support
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
        nbinslogmass = int(sys.argv[3])
    except:
        nbinslogmass = 51

    try:
        nbinslambda = int(sys.argv[4])
    except:
        nbinslambda = 81
    
    try:
        numcores = int(sys.argv[5])
    except:
        numcores = 8
        
    try:
        calcirfmatrices = int(sys.argv[6])
    except:
        calcirfmatrices = 0
    
    # Setting up axis values so it doesn't re-run in any parallelisation setups
    signalspecfunc = darkmatterdoubleinput

    lontrue_mesh_nuisance, logetrue_mesh_nuisance, lattrue_mesh_nuisance = np.meshgrid(spatialaxistrue, log10eaxistrue, spatialaxistrue)

    
    
    currentdirecyory = os.getcwd()
    stemdirectory = currentdirecyory+f'/data/{identifier}'
    print("\nstem directory: ", stemdirectory, '\n')

    rundirectory = f'{stemdirectory}/{runnum}'
    
    print("\nrun directory: ", rundirectory, '\n')
    
    
    signal_log10e_measured,  signal_lon_measured, signal_lat_measured = np.load(f"{rundirectory}/meassigsamples.npy")
    bkg_log10e_measured, bkg_lon_measured, bkg_lat_measured = np.load(f"{rundirectory}/measbkgsamples.npy")
    
    truelambda, Nsamples, truelogmassval = np.load(f"{rundirectory}/params.npy")[1,:]
    truelambda = float(truelambda)
    Nsamples = int(Nsamples)
    
    logbkgpriorvalues = np.load(f"{rundirectory}/logbackgroundprior.npy")

    params              = np.load(f"{rundirectory}/params.npy")
    
    params[1,:]         = params[1,:]
    truelogmass         = float(params[1,2])
    truelambdaval       = float(params[1,0])
            
    
        
    measured_log10e = list(signal_log10e_measured)+list(bkg_log10e_measured)
    measured_lon = list(signal_lon_measured)+list(bkg_lon_measured)
    measured_lat = list(signal_lat_measured)+list(bkg_lat_measured)

    if calcirfmatrices:
        lontrue_mesh_psf, logetrue_mesh_psf, lattrue_mesh_psf, lonrecon_mesh_psf, latrecon_mesh_psf = np.meshgrid(spatialaxistrue, log10eaxistrue, spatialaxistrue, spatialaxis, spatialaxis)
        psfmatrix = psf(np.array([lonrecon_mesh_psf.flatten(), latrecon_mesh_psf.flatten()]), np.array([lontrue_mesh_psf.flatten(), lattrue_mesh_psf.flatten()]), logetrue_mesh_psf.flatten()).reshape(logetrue_mesh_psf.shape)
        psfnormalisation  = special.logsumexp(psfmatrix, axis=(-2,-1))
        
        

            
        lontrue_mesh_edisp, logetrue_mesh_edisp, lattrue_mesh_edisp, logerecon_mesh_edisp,  = np.meshgrid(spatialaxistrue, log10eaxistrue, spatialaxistrue, log10eaxis)
        edispmatrix = edisp(logerecon_mesh_edisp.flatten(), logetrue_mesh_edisp.flatten(), np.array([lontrue_mesh_edisp.flatten(), lattrue_mesh_edisp.flatten()])).reshape(logetrue_mesh_edisp.shape)
        edispnormalisation  = special.logsumexp(edispmatrix+logjacob, axis=-1)


        edispnormalisation[edispnormalisation==-np.inf] = 0
        psfnormalisation[psfnormalisation==-np.inf] = 0   
        
        
        
        edispmatrix = edispmatrix-edispnormalisation[:,:,:,np.newaxis]
        psfmatrix = psfmatrix-psfnormalisation[:,:,:,np.newaxis, np.newaxis]
        
        
        np.save("edispmatrix.npy", edispmatrix)
        np.save("psfmatrix.npy", psfmatrix)
        np.save("edispnormalisation.npy", edispnormalisation)
        np.save("psfnormalisation.npy", psfnormalisation)
    else:
        edispmatrix = np.load("edispmatrix.npy")
        psfmatrix = np.load("psfmatrix.npy")

    irfindexlist = []

    
    with Pool(numcores) as pool: 
            
        irfindexlist =  pool.map(calcrirfindices, tqdm(zip(measured_log10e, np.array([measured_lon, measured_lat]).T), total=len(list(measured_log10e)), ncols=100, desc="Calculating irf values"))

        pool.close() 
    
    irfindexlist = np.array(irfindexlist)
    
    
    
    # Loading in values
    
    psfnormalisation  = special.logsumexp(psfmatrix, axis=(-2,-1))
    edispnormalisation  = special.logsumexp(edispmatrix+logjacob, axis=-1)


    edispnormalisation[edispnormalisation==-np.inf] = 0
    psfnormalisation[psfnormalisation==-np.inf] = 0   
    
    
    
    edispmatrix = edispmatrix-edispnormalisation[:,:,:,np.newaxis]
    psfmatrix = psfmatrix-psfnormalisation[:,:,:,np.newaxis, np.newaxis]
    
    
    
    
    
    # logbackgroundprior = np.load(f"{firstrundirectory}/logbackgroundprior.npy")
    truelambda, Nsamples, truelogmassval = np.load(f"{rundirectory}/params.npy")[1,:]
    Nsamples = int(Nsamples)
    truelogmassval = float(truelogmassval)
    truelambda = float(truelambda)


    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # Signal Nuisance Parameter Marginalisation


    sigdistsetup = setup_full_fake_signal_dist
    # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
    np.seterr(divide='ignore', invalid='ignore')




    nsig = int(round(truelambda*Nsamples))


    # Generating the range of log mass values to be tested
    logmasswindowwidth      = 5/np.sqrt(nsig)

    logmasslowerbound       = truelogmassval-logmasswindowwidth
    logmassupperbound       = truelogmassval+logmasswindowwidth

    if logmasslowerbound<log10eaxis[0]:
        logmasslowerbound = log10eaxis[0]
    if logmassupperbound>2:
        logmassupperbound = 2
        
    logmassrange            = np.linspace(logmasslowerbound, logmassupperbound, nbinslogmass) 


    # Generating the range of lambda values to be tested
    lambdawindowwidth      = 5/np.sqrt(Nsamples)
    
    lambdalowerbound       = truelambda-lambdawindowwidth
    lambdaupperbound       = truelambda+lambdawindowwidth
    

    if lambdalowerbound<0:
        lambdalowerbound = 0
    if lambdaupperbound>1:
        lambdaupperbound = 1

    lambdarange            = np.linspace(lambdalowerbound, lambdaupperbound, nbinslambda) 
    
    
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # Nuisance Parameter Marginalisation with Background Prior
    
    bkgpriorarray  = bkgdist(logetrue_mesh_nuisance, lontrue_mesh_nuisance, lattrue_mesh_nuisance)
    bkgpriorarray = bkgpriorarray.T - special.logsumexp(bkgpriorarray.T+logjacobtrue)
    # print(special.logsumexp(bkgpriorarray.T+logjacobtrue).shape)

    bkgmargvals = special.logsumexp(logjacobtrue+bkgpriorarray+edispmatrix[:,:,:,irfindexlist[:,0]].T+psfmatrix[:,:,:,irfindexlist[:,1],irfindexlist[:,2]].T, axis=(1,2,3))
    bkgmargvals = np.array(bkgmargvals)
    
    print(f"Shape of array containing the results of marginalising the nuisance parameters with the background prior: {bkgmargvals.shape}")
    
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # Nuisance Parameter Marginalisation with Signal Prior
    
    
    
    print(time.strftime("Current time is %d of %b, at %H:%M:%S"))

    sigmargresults = []

    for logmass in tqdm(logmassrange, total=logmassrange.shape[0]):
        
        priorvals= setup_full_fake_signal_dist(logmass, specfunc=signalspecfunc)(logetrue_mesh_nuisance, lontrue_mesh_nuisance, lattrue_mesh_nuisance)
        
        priornormalisation = special.logsumexp(priorvals.T+logjacobtrue)
        
        priorvals = priorvals.T-priornormalisation
        
        
        tempsigmargfunc = functools.partial(marginalisenuisance, prior=priorvals, edispmatrix=edispmatrix, psfmatrix=psfmatrix)
        result = [tempsigmargfunc(singleeventindices) for singleeventindices in irfindexlist]
        
        sigmargresults.append(result)
    signal_log_marginalisationvalues = np.array(sigmargresults)


        
    print(len(irfindexlist))
    print(Nsamples)
    
    assert len(irfindexlist) == int(Nsamples)
    assert irfindexlist.ndim==2
        
    np.save(f'{rundirectory}/lambdarange_direct.npy',lambdarange)
    np.save(f'{rundirectory}/logmassrange_direct.npy',logmassrange)
    np.save(f"{rundirectory}/irfindexlist.npy", irfindexlist)
    np.save(f"{rundirectory}/log_signal_marginalisations.npy", signal_log_marginalisationvalues)
    np.save(f"{rundirectory}/log_background_marginalisations.npy", bkgmargvals)