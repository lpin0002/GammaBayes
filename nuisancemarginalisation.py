from utils3d import *
from BFCalc.createspectragrids import darkmatterdoubleinput, energymassinputspectralfunc
import numpy as np
from scipy import special
from tqdm import tqdm
import os, sys, time, functools
from multiprocessing import Pool, freeze_support
sys.path.append("BFCalc")


def sigmarg(logmass, specfunc, irfindexlist, edispmatrix, psfmatrix, logetrue_mesh_nuisance, lontrue_mesh_nuisance, lattrue_mesh_nuisance, logjacobtrue=logjacobtrue):
    priorvals= setup_full_fake_signal_dist(logmass, specfunc=specfunc)(logetrue_mesh_nuisance, lontrue_mesh_nuisance, lattrue_mesh_nuisance)
        
    priornormalisation = special.logsumexp(priorvals.T+logjacobtrue)
    
    priorvals = priorvals.T-priornormalisation
    
    
    tempsigmargfunc = functools.partial(marginalisenuisance, prior=priorvals, edispmatrix=edispmatrix, psfmatrix=psfmatrix)
    result = [tempsigmargfunc(singleeventindices) for singleeventindices in irfindexlist]
    
    return result

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
        nbinslogmass = 101

    try:
        nbinslambda = int(sys.argv[5])
    except:
        nbinslambda = 161
    
    try:
        numcores = int(sys.argv[6])
    except:
        numcores = 1
        
    try:
        calcirfmatrices = int(sys.argv[7])
    except:
        calcirfmatrices = 0
    np.seterr(divide = 'ignore')

    
    # Setting up axis values so it doesn't re-run in any parallelisation setups
    signalspecfunc = energymassinputspectralfunc

    lontrue_mesh_nuisance, logetrue_mesh_nuisance, lattrue_mesh_nuisance = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue)

    
    
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

    
    # If these files do not exist run the setup.py file
    edispmatrix = np.load("edispmatrix.npy")
    psfmatrix = np.load("psfmatrix.npy")

    irfindexlist = []

    
    with Pool(numcores) as pool: 
            
        for result in tqdm(pool.imap(calcrirfindices, zip(measured_log10e, np.array([measured_lon, measured_lat]).T)), total=len(list(measured_log10e)), ncols=100, desc="Calculating irf values..."):
                irfindexlist.append(result)

        pool.close() 
        
    irfindexlist = np.array(irfindexlist)
    print("Done with irf calculations\n")
    
    
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




    nsig = int(round(truelambda*Nsamples*totalnumberofruns))


    # Generating the range of log mass values to be tested
    logmasswindowwidth      = 3/np.sqrt(nsig)

    logmasslowerbound       = truelogmassval-logmasswindowwidth
    logmassupperbound       = truelogmassval+logmasswindowwidth

    if logmasslowerbound<log10eaxis[0]:
        logmasslowerbound = log10eaxis[0]
    if logmassupperbound>2:
        logmassupperbound = 2
        
    logmassrange            = np.linspace(logmasslowerbound, logmassupperbound, nbinslogmass) 


    # Generating the range of lambda values to be tested
    lambdawindowwidth      = 3/np.sqrt(Nsamples*totalnumberofruns)
    
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
    print("Calculating the marginalisation with the background model...")

    bkgpriorarray  = logbkgpriorvalues
    bkgpriorarray = bkgpriorarray.T - special.logsumexp(bkgpriorarray.T+logjacobtrue)
    # print(special.logsumexp(bkgpriorarray.T+logjacobtrue).shape)

    bkgmargvals = []
    for irfmatrixcoord in tqdm(irfindexlist, ncols=100):
        bkgmargvals.append(special.logsumexp(logjacobtrue+bkgpriorarray+edispmatrix[:,:,:,irfmatrixcoord[0]].T+psfmatrix[:,:,:,irfmatrixcoord[1],irfmatrixcoord[2]].T))
    
    bkgmargvals = np.array(bkgmargvals)
    
    print(f"Shape of array containing the results of marginalising the nuisance parameters with the background prior: {bkgmargvals.shape}")
    
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # Nuisance Parameter Marginalisation with Signal Prior
    
    
    print("\nCalculating the marginalisation with the signal model...")

    print(time.strftime("Current time is %d of %b, at %H:%M:%S"))

    sigmargresults = []

    # for logmass in tqdm(logmassrange, total=logmassrange.shape[0], ncols=100):
        
    #     priorvals= setup_full_fake_signal_dist(logmass, specfunc=signalspecfunc)(logetrue_mesh_nuisance, lontrue_mesh_nuisance, lattrue_mesh_nuisance)
        
    #     priornormalisation = special.logsumexp(priorvals.T+logjacobtrue)
        
    #     priorvals = priorvals.T-priornormalisation
        
        
    #     tempsigmargfunc = functools.partial(marginalisenuisance, prior=priorvals, edispmatrix=edispmatrix, psfmatrix=psfmatrix)
    #     result = [tempsigmargfunc(singleeventindices) for singleeventindices in irfindexlist]
        
    #     sigmargresults.append(result)
    
    sigmarg_partial = functools.partial(sigmarg, specfunc=signalspecfunc, irfindexlist=irfindexlist, edispmatrix=edispmatrix, psfmatrix=psfmatrix,
                                        logetrue_mesh_nuisance=logetrue_mesh_nuisance, lontrue_mesh_nuisance=lontrue_mesh_nuisance, 
                                        lattrue_mesh_nuisance=lattrue_mesh_nuisance, logjacobtrue=logjacobtrue)
    with Pool(numcores) as pool:
        sigmargresults = pool.map(sigmarg_partial, tqdm(logmassrange, ncols=100, total=logmassrange.shape[0]))
    print(time.strftime("Current time is %d of %b, at %H:%M:%S"))
    print("Finished signal marginalisation. \nNow converting to numpy arrays and saving the result...")
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