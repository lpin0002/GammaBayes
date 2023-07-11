from BFCalc.createspectragrids import darkmatterdoubleinput
from utils3d import *
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
        nbinslogmass = int(sys.argv[2])
    except:
        nbinslogmass = 11

    try:
        nbinslambda = int(sys.argv[3])
    except:
        nbinslambda = 11
    
    try:
        numcores = int(sys.argv[4])
    except:
        numcores = 10
        
    signalspecfunc = darkmatterdoubleinput
    lontrue_mesh_nuisance, logetrue_mesh_nuisance, lattrue_mesh_nuisance = np.meshgrid(spatialaxistrue, log10eaxistrue, spatialaxistrue)


    
    
    
    #Generating various paths to files within stem directory
    currentdirecyory = os.getcwd()
    stemdirectory = currentdirecyory+f'/data/{identifier}'
    print("\nstem directory: ", stemdirectory, '\n')
    
    rundirs = [x[0] for x in os.walk(stemdirectory)][1:]
    print("number of run directories: ", len(rundirs), '\n')
    
    firstrundirectory = rundirs[0]
    
    
    
    # Loading in values
    irfproblist = np.load(f"{stemdirectory}/irfproblist.npy")
    # logbackgroundprior = np.load(f"{firstrundirectory}/logbackgroundprior.npy")
    truelambda, Nsamples, truelogmassval = np.load(f"{firstrundirectory}/params.npy")[1,:]
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
    # Nuisance Parameter Marginalisation with Signal Prior

    tempsigmargfunction = functools.partial(evaluateformass, irfvals=irfproblist, specfunc=signalspecfunc, lontrue_mesh_nuisance=lontrue_mesh_nuisance, logetrue_mesh_nuisance=logetrue_mesh_nuisance, lattrue_mesh_nuisance=lattrue_mesh_nuisance )

    signal_log_marginalisationvalues = []
    
    start = time.perf_counter()

    with Pool(10) as pool: 
            
        signal_log_marginalisationvalues = pool.map(tempsigmargfunction, logmassrange)

        pool.close() 
        
    print(time.perf_counter()-start)
        
    signal_log_marginalisationvalues = np.array(signal_log_marginalisationvalues)
    print(f"Shape of array containing the results of marginalising the nuisance parameters with the signal prior: {signal_log_marginalisationvalues.shape}")
    
    
    
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # Nuisance Parameter Marginalisation with Background Prior
    
    bkgpriorarray  = bkgdist(logetrue_mesh_nuisance, lontrue_mesh_nuisance, lattrue_mesh_nuisance)
    bkgpriorarray = bkgpriorarray - special.logsumexp(bkgpriorarray.T+logjacobtrue)

    bkgmargvals = [special.logsumexp(bkgpriorarray.T+irfarray.T+logjacobtrue) for irfarray in irfproblist]
    bkgmargvals = np.array(bkgmargvals)
    
    print(f"Shape of array containing the results of marginalising the nuisance parameters with the background prior: {bkgmargvals.shape}")

    
    
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # Combine nuisance parameter marginalisation results to calculate (log) posterior
    
    

    log_posterior = []

    for lambdaval in tqdm(lambdarange, total=lambdarange.shape[0]):
        log_posterior.append([np.sum(np.logaddexp(np.log(lambdaval)+signal_log_marginalisationvalues[logmassindex,:], np.log(1-lambdaval)+bkgmargvals)) for logmassindex in range(len(list(logmassrange)))])


    unnormalised_log_posterior = np.array(log_posterior)


    print(f"Shape of the array containing the log of the unnormalised posterior: {unnormalised_log_posterior.shape}")

    
    
    
    np.save(f'{stemdirectory}/lambdarange_direct.npy',lambdarange)
    np.save(f'{stemdirectory}/logmassrange_direct.npy',logmassrange)
    np.save(f'{stemdirectory}/unnormalised_logposterior_direct.npy', unnormalised_log_posterior)
    






