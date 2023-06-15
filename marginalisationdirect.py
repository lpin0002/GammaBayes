
from utils import log10eaxis, bkgdist, makedist, edisp,logjacob, evaluateintegral, evaluateformass, setup_full_fake_signal_dist, calcirfvals
from scipy import special
import numpy as np
import os, time, sys
from tqdm import tqdm
from BFCalc.BFInterp import DM_spectrum_setup
import functools
from multiprocessing import Pool, freeze_support
import multiprocessing




if __name__ == '__main__':
       try:
              identifier = sys.argv[1]
       except:
              identifier = time.strftime("%d%m%H")
       try:
              runnum = sys.argv[2]
       except:
              runnum = 1

       try:
              nbinslogmass = int(sys.argv[3])
       except:
              nbinslogmass = 21
       try:
              nbinslambda = int(sys.argv[4])
       except:
              nbinslambda = 21
       
       sigdistsetup = setup_full_fake_signal_dist
       # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
       np.seterr(divide='ignore', invalid='ignore')

       sigsamples           = np.load(f"data/{identifier}/{runnum}/truesigsamples.npy")
       sigsamples_measured  = np.load(f"data/{identifier}/{runnum}/meassigsamples.npy")
       
       sig_log10e_samples = sigsamples[0]
       sig_offset_samples = sigsamples[1]
       
       sigsamples_log10e_measured = sigsamples_measured[0]
       sigsamples_offset_measured = sigsamples_measured[1]

       # bkgsamples           = np.load(f"data/{identifier}/{runnum}/truebkgsamples.npy")
       # bkgsamples_measured  = np.load(f"data/{identifier}/{runnum}/measbkgsamples.npy")
       bkgsamples = []
       bkgsamples_measured = []
       params               = np.load(f"data/{identifier}/{runnum}/params.npy")
       params[1,:]          = params[1,:]
       truelogmass          = float(params[1,2])
       nevents              = int(params[1,1])
       true_offset_vals             = np.array(list(sig_offset_samples)+list([]))
       measured_offset_vals         = np.array(list(sigsamples_offset_measured)+list([]))
       true_log10e_vals             = np.array(list(sig_log10e_samples)+list([]))
       measured_log10e_vals         = np.array(list(sigsamples_log10e_measured)+list([]))

       logmasswindowwidth   = 3/np.sqrt(nevents)
       logmasslowerbound    = truelogmass-logmasswindowwidth
       logmassupperbound    = truelogmass+logmasswindowwidth

       
       if logmasslowerbound<-1.00:
              logmasslowerbound = -1.00
       if logmassupperbound>2:
              logmassupperbound = 2
       

       logmassrange         = np.linspace(logmasslowerbound,logmassupperbound,nbinslogmass)
       
       np.save(f'data/{identifier}/{runnum}/logmassrange_direct.npy',logmassrange)
       
       irfvals = [calcirfvals(logemeasured, offsetmeasured) for logemeasured, offsetmeasured in tqdm(zip(sigsamples_log10e_measured, sigsamples_offset_measured), 
                                                                                                     total=len(list(sigsamples_log10e_measured)),
                                                                                                     desc='Calculating irfvals', ncols=100,)]
       
       produce_posterior_function = functools.partial(evaluateformass, irfvals=irfvals)
       logmass_logposterior = []
       with Pool(10) as pool: 
              
              for result in tqdm(pool.imap(produce_posterior_function, logmassrange), total=len(list(logmassrange)), ncols=100, desc="Calculating signal marginalisations..."):
                     logmass_logposterior.append(result)

              pool.close() 
           
       print("Done calculating the posterior")
       np.save(f'data/{identifier}/logmass_logposterior_direct.npy', logmass_logposterior)









