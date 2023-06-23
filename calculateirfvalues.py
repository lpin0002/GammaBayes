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
        numcores = int(sys.argv[2])
    except:
        numcores = 10
    
    
    
    
    currentdirecyory = os.getcwd()
    stemdirectory = currentdirecyory+f'/data/{identifier}'
    print("\nstem directory: ", stemdirectory, '\n')

    rundirs = [x[0] for x in os.walk(stemdirectory)][1:]
    print("number of run directories: ", len(rundirs), '\n')
    
    
    
    firstrundirectory = rundirs[0]
    
    print(firstrundirectory)
    
    
    signal_log10e_measured,signal_offset_measured = np.load(f"{firstrundirectory}/meassigsamples.npy")
    bkg_log10e_measured,bkg_offset_measured = np.load(f"{firstrundirectory}/measbkgsamples.npy")
    
    truelambda, Nsamples, truelogmassval = np.load(f"{firstrundirectory}/params.npy")[1,:]
    truelambda = float(truelambda)
    totalevents = int(Nsamples)
    
    
    for rundir in rundirs[1:]:
        runnum = rundir.replace(stemdirectory+'/', '')
        print("runnum: ", runnum)
        params              = np.load(f"{rundir}/params.npy")
        signal_log10e_measuredtemp,signal_offset_measuredtemp = np.load(f"{rundir}/meassigsamples.npy")
        bkg_log10e_measuredtemp,bkg_offset_measuredtemp = np.load(f"{rundir}/measbkgsamples.npy")
        
        
        signal_log10e_measured = np.concatenate((signal_log10e_measured, signal_log10e_measuredtemp))
        signal_offset_measured = np.concatenate((signal_offset_measured, signal_offset_measuredtemp))
        bkg_log10e_measured = np.concatenate((bkg_log10e_measured, bkg_log10e_measuredtemp))
        bkg_offset_measured = np.concatenate((bkg_offset_measured, bkg_offset_measuredtemp))
        
        params[1,:]         = params[1,:]
        truelogmass         = float(params[1,2])
        nevents             = int(params[1,1])
        totalevents         +=nevents
        truelambdaval       = float(params[1,0])
            
            
    measured_log10e_vals    = list(signal_log10e_measured)+list(bkg_log10e_measured)
    measured_offset_vals    = list(signal_offset_measured)+list(bkg_offset_measured)
    
    

    testvalue = calcirfvals([measured_log10e_vals[0], measured_offset_vals[0]])
    print(testvalue)
    irfvals = []
    with Pool(numcores) as pool: 
            for result in tqdm(pool.imap(calcirfvals, zip(measured_log10e_vals, measured_offset_vals)), 
                                total=len(list(measured_log10e_vals)), ncols=100, desc="Calculating irfvals"):
                    irfvals.append(result)

            pool.close() 
            
    np.save(f"{stemdirectory}/irfvalues.npy", irfvals)