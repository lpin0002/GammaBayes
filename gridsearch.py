from utils import log10eaxis, bkgdist, makedist, edisp,logjacob, evaluateintegral, evaluateformass, setup_full_fake_signal_dist, calcirfvals
from scipy import special
import numpy as np
import os, time, sys
from tqdm import tqdm
from BFCalc.BFInterp import DM_spectrum_setup
import functools
from multiprocessing import Pool, freeze_support
import multiprocessing


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
    
    
    
    # Generating various paths to files within stem directory
    currentdirecyory = os.getcwd()
    stemdirectory = currentdirecyory+f'/data/{identifier}'
    print("\nstem directory: ", stemdirectory, '\n')
    
    rundirs = [x[0] for x in os.walk(stemdirectory)][1:]
    print("number of run directories: ", len(rundirs), '\n')
    
    firstrundirectory = rundirs[0]
    
    
    
    # Loading in values
    irfvals = np.load(f"{stemdirectory}/irfvalues.npy")
    bkgbinnedprior = np.load(f"{firstrundirectory}/backgroundprior.npy")
    truelambda, Nsamples, truelogmassval = np.load(f"{firstrundirectory}/params.npy")[1,:]
    truelogmassval = float(truelogmassval)
    truelambda = float(truelambda)
    totalevents = int(Nsamples)*len(rundirs)
    
    
    
    # Creating the range of values for logmass and lambda based on the number of bins inputted
        # and the number of total events
    
    nsig = int(np.round(truelambda*totalevents))

    logmasswindowwidth      = 2/np.sqrt(nsig)
    lambdawindowwidth       = 4/np.sqrt(totalevents)

    logmasslowerbound       = truelogmassval-logmasswindowwidth
    logmassupperbound       = truelogmassval+logmasswindowwidth

    lambdalowerbound        = truelambda-lambdawindowwidth
    lambdaupperbound        = truelambda+lambdawindowwidth


    if logmasslowerbound<log10eaxis[1]:
            logmasslowerbound = log10eaxis[1]
    if logmassupperbound>2:
            logmassupperbound = 2
            
            
    if lambdalowerbound<0:
            lambdalowerbound = 0
    if lambdaupperbound>1:
            lambdaupperbound = 1


    logmassrange            = np.linspace(logmasslowerbound, logmassupperbound, nbinslogmass)    
    lambdarange             = np.linspace(lambdalowerbound, lambdaupperbound, nbinslambda)


    produce_logsigmarg_function = functools.partial(evaluateformass, irfvals=irfvals, specsetup=DM_spectrum_setup)
    signal_log_marginalisationvalues = []
    with Pool(numcores) as pool: 
            
            for result in tqdm(pool.imap(produce_logsigmarg_function, logmassrange), total=len(list(logmassrange)), ncols=100, desc="Calculating signal marginalisations..."):
                    signal_log_marginalisationvalues.append(result)

            pool.close() 
    signal_log_marginalisationvalues = np.array(signal_log_marginalisationvalues)
    
    
    
    produce_logbkgmarg_function = functools.partial(evaluateintegral, priorvals=bkgbinnedprior-logjacob)

    bkg_log_marginalisationvalues = []
    with Pool(numcores) as pool: 
            for result in tqdm(pool.imap(produce_logbkgmarg_function, irfvals), total=len(list(irfvals)), ncols=100, desc="Calculating background marginalisations..."):
                    bkg_log_marginalisationvalues.append(result)

            pool.close() 
    bkg_log_marginalisationvalues = np.array(bkg_log_marginalisationvalues)


    logposterior = []

    for ii, logmass in tqdm(enumerate(logmassrange), total=len(list(logmassrange))):
            singlerow = []
            for jj, lambdaval in enumerate(lambdarange):
                    product = np.sum(np.logaddexp(np.log(lambdaval)+signal_log_marginalisationvalues[ii,:], np.log(1-lambdaval)+bkg_log_marginalisationvalues))
                    singlerow.append(product)
            logposterior.append(singlerow)
    logposterior = np.array(logposterior)

    normalisedlogposterior = logposterior-special.logsumexp(logposterior)
    
    
    np.save(f'{stemdirectory}/lambdarange_direct.npy',lambdarange)
    np.save(f'{stemdirectory}/logmassrange_direct.npy',logmassrange)
    np.save(f'{stemdirectory}/logposterior_direct.npy', logposterior)
    np.save(f'{stemdirectory}/normalised_logposterior_direct.npy', normalisedlogposterior)





