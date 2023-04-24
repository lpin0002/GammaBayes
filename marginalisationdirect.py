
from utils import inverse_transform_sampling, axis, bkgdist, makedist, edisp, eaxis_mod, COLOR,logjacob
from scipy import integrate, special, interpolate, stats
import numpy as np
import os, time, random, warnings, concurrent.futures, sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import chime
from BFCalc.BFInterp import DM_spectrum_setup
import functools
from multiprocessing import Pool, freeze_support
import multiprocessing
# chime.info('sonic')





def sigmarg(logmass, edisplist, sigdistsetup, measuredvals, logjacob=logjacob, axis=axis):
       tempsigdist = sigdistsetup(logmass)
       tempmarglogmassrow = []
       tempsigdist = sigdistsetup(logmass)
       tempsigdistaxis = tempsigdist(axis) - special.logsumexp(tempsigdist(axis)+logjacob)
       for i, sample in enumerate(measuredvals):
              tempsigmarg = special.logsumexp(tempsigdistaxis+edisplist[i]+logjacob)
              tempmarglogmassrow.append(tempsigmarg)
       return np.array(tempmarglogmassrow)


def sigmargwrapper(logmass, edisplist, sigdistsetup, measuredvals):
       return sigmarg(logmass, edisplist, sigdistsetup, measuredvals)

def posteriorcalc(lambdaval, sigmarglogzvals, bkgmarglist, measuredvals):
       tempmargval = 0
       for i, sample in enumerate(measuredvals):
              tempmargval += np.logaddexp(np.log(lambdaval)+sigmarglogzvals[j][i],np.log(1-lambdaval)+bkgmarglist[i])
       # print(f"{tempmargval:.2e}", end='\r')
              
       return tempmargval

if __name__ == '__main__':
       try:
              identifier = sys.argv[1]
       except:
              identifier = time.strftime("%d%m%H")

       try:
              nbinslogmass = int(sys.argv[2])
       except:
              nbinslogmass = 21
       try:
              nbinslambda = int(sys.argv[3])
       except:
              nbinslambda = 21
       sigdistsetup = makedist
       # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
       np.seterr(divide='ignore', invalid='ignore')

       sigsamples           = np.load(f"data/{identifier}/truesigsamples.npy")
       sigsamples_measured  = np.load(f"data/{identifier}/meassigsamples.npy")
       bkgsamples           = np.load(f"data/{identifier}/truebkgsamples.npy")
       bkgsamples_measured  = np.load(f"data/{identifier}/measbkgsamples.npy")
       params               = np.load(f"data/{identifier}/params.npy")
       params[1,:]          = params[1,:]
       truelogmass          = float(params[1,2])
       nevents              = int(params[1,1])
       truelambdaval        = float(params[1,0])
       truevals             = np.concatenate((sigsamples, bkgsamples))
       measuredvals         = np.concatenate((sigsamples_measured,bkgsamples_measured))

       logmasswindowwidth   = 2/np.sqrt(nevents)
       logmasslowerbound    = truelogmass-logmasswindowwidth
       logmassupperbound    = truelogmass+logmasswindowwidth

       lambdavalwindowwidth = 5/np.sqrt(nevents)
       lambdalowerbound     = truelambdaval-lambdavalwindowwidth
       lambdaupperbound     = truelambdaval+lambdavalwindowwidth
       if logmasslowerbound<axis[0]:
              logmasslowerbound = axis[1]
       if lambdaupperbound>1.:
              lambdaupperbound=1
       if lambdalowerbound<0:
              lambdalowerbound=0
       

       logmassrange         = np.linspace(logmasslowerbound,logmassupperbound,nbinslogmass)
       lambdarange          = np.linspace(lambdalowerbound,lambdaupperbound,nbinslambda)
       # logmassrange = np.linspace(axis[1],axis[-1],nbins)
       # lambdarange = np.linspace(0,1,nbins)
       np.save(f'data/{identifier}/logmassrange_Direct.npy',logmassrange)
       np.save(f'data/{identifier}/lambdarange_Direct.npy',lambdarange)
       # lambdarange = np.array([0.45, 0.5])
       print("logmassrange: ", logmassrange[0], logmassrange[-1])
       print("lambdarange: ", lambdarange[0], lambdarange[-1])

       print(COLOR.BOLD+f"""\n\n{COLOR.BOLD}{COLOR.GREEN}IMPORTANT PARAMETERS: {COLOR.END}
       {COLOR.YELLOW}number of events{COLOR.END} being analysed/were simulated is {nevents:.1e}. 

       {COLOR.YELLOW}true log mass value{COLOR.END} used for the signal model is {truelogmass} or equivalently a mass of roughly {np.round(np.power(10., truelogmass),3):.2e}.

       {COLOR.YELLOW}fraction of signal events to total events{COLOR.END} is {truelambdaval}.

       {COLOR.YELLOW}bounds for the log energy range{COLOR.END} are {axis[0]:.2e} and {axis[-1]:.2e} translating into energy bounds of {np.power(10.,axis[0]):.2e} and {np.power(10.,axis[-1]):.2e}.

       {COLOR.YELLOW}bounds for the log mass range [TeV]{COLOR.END} are {logmassrange[0]:.2e} and {logmassrange[-1]:.2e} translating into mass bounds of {np.power(10.,logmassrange[0]):.2e} and {np.power(10.,logmassrange[-1]):.2e} [TeV].

       {COLOR.YELLOW}bounds for the lambda range{COLOR.END} are {lambdarange[0]:.2e} and {lambdarange[-1]:.2e}.

       \n""")

       edispnorms = np.array([special.logsumexp(edisp(axis,axisval)+logjacob) for axisval in axis])

       if -np.inf in edispnorms:
              print(COLOR.BOLD+"Your energy dispersion normalisation has -np.inf inside, which will almostly definitely mean your energy dispersion or the normalisation is wrong."+COLOR.END)

       edisplist = []
       bkgmarglist = []
       bkgdistnormed = bkgdist(axis) - special.logsumexp(bkgdist(axis)+logjacob)


       print(f"There are {COLOR.BLUE}{nevents}{COLOR.END} events being analyzed.")
       for i, sample in tqdm(enumerate(measuredvals),desc="Calculating edisp vals and bkg marginalisation", ncols=100):
              edisplist.append(edisp(sample,axis)-edispnorms)
              bkgmarglist.append(special.logsumexp(bkgdistnormed+edisplist[i]+logjacob))
       edisplist = np.array(edisplist)
       
       np.save(f'data/{identifier}/edisplist_Direct.npy', edisplist)
       np.save(f'data/{identifier}/bkgmarglist_Direct.npy', bkgmarglist)


       sigmarglogzvals = []
       num_cores = multiprocessing.cpu_count()
       print(f"You have {num_cores} cores on your machine")
       with Pool(num_cores) as pool:
              func = functools.partial(sigmargwrapper, edisplist=edisplist, sigdistsetup=sigdistsetup, measuredvals=measuredvals)
            
              for result in tqdm(pool.imap(func, logmassrange), total=len(list(logmassrange)), ncols=100, desc="Calculating signal marginalisations..."):
                     sigmarglogzvals.append(result)

              pool.close()
       print("Done calculating the signal marginalisations.")
       np.save(f'data/{identifier}/sigmarglogzvals_Direct.npy', sigmarglogzvals)


       chime.info('sonic')







