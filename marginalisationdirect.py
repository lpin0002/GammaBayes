
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
              numcores = int(sys.argv[3])
       except:
              numcores = 10
              
       try:
              nbinslogmass = int(sys.argv[4])
       except:
              nbinslogmass = 11

       try:
              nbinslambda = int(sys.argv[5])
       except:
              nbinslambda = 11
              
       
       
              
       
       
       
       siglogevals, sigoffsetvals = np.load(f"data/{identifier}/{runnum}/truesigsamples.npy")
       signal_log10e_measured,signal_offset_measured = np.load(f"data/{identifier}/{runnum}/meassigsamples.npy")
       bkglogevals,bkgoffsetvals = np.load(f"data/{identifier}/{runnum}/truebkgsamples.npy")
       bkg_log10e_measured,bkg_offset_measured = np.load(f"data/{identifier}/{runnum}/measbkgsamples.npy")
       bkgbinnedprior = np.load(f"data/{identifier}/{runnum}/backgroundprior.npy")
       
       truelambda, Nsamples, truelogmassval = np.load(f"data/{identifier}/{runnum}/params.npy")[1,:]
       truelambda = float(truelambda)
       Nsamples = int(Nsamples)
       truelogmassval = float(truelogmassval)
       
       
       
       
       true_offset_vals             = np.array(list(sigoffsetvals)+list(bkgoffsetvals))
       measured_offset_vals         = np.array(list(signal_offset_measured)+list(bkg_offset_measured))
       true_log10e_vals             = np.array(list(siglogevals)+list(bkglogevals))
       measured_log10e_vals         = np.array(list(signal_log10e_measured)+list(bkg_log10e_measured))

       nsig = int(np.round(truelambda*Nsamples))

       logmasswindowwidth      = 2/np.sqrt(nsig)

       lambdawindowwidth       = 4/np.sqrt(Nsamples)


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
       
       testvalue = calcirfvals([measured_log10e_vals[0], measured_offset_vals[0]])
       print(testvalue)
       irfvals = []
       with Pool(numcores) as pool: 
              for result in tqdm(pool.imap(calcirfvals, zip(measured_log10e_vals, measured_offset_vals)), 
                                   total=len(list(measured_log10e_vals)), ncols=100, desc="Calculating irfvals"):
                     irfvals.append(result)

              pool.close() 
              
              
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
       
       np.save(f'data/{identifier}/{runnum}/lambdarange_direct.npy',lambdarange)
       np.save(f'data/{identifier}/{runnum}/logmassrange_direct.npy',logmassrange)
       np.save(f'data/{identifier}/logposterior_direct.npy', logposterior)
       np.save(f'data/{identifier}/normalised_logposterior_direct.npy', normalisedlogposterior)









