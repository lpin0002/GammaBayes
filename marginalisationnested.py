
from utils import inverse_transform_sampling, axis, bkgdist, makedist, edisp, eaxis_mod, COLOR,logjacob, logpropdist
from rundynesty import rundynesty
from scipy import integrate, special, interpolate, stats
import numpy as np
import os, time, random, warnings, concurrent.futures, sys
from tqdm import tqdm
import matplotlib.pyplot as plt
# import chime
from BFCalc.BFInterp import DM_spectrum_setup
import functools
from multiprocessing import Pool, freeze_support
import multiprocessing
# chime.info('sonic')

log10eaxis = axis

#### Inputs for script in correct order an type are
# identifier (whatever type you want), run number (int), nevents (int), true log_10 mass (float), true lambda val (float), marg cores (int)


def marg(index, edisplist, dist, log10eaxis=log10eaxis, print_progress=False):
       margsampler = rundynesty(dist, edisplist[index], log10eaxis=log10eaxis, print_progress=print_progress)
       return margsampler


def margwrapper(index, edisplist, dist, log10eaxis, print_progress):
       return marg(index, edisplist, dist, log10eaxis, print_progress)


if __name__ == '__main__':
       np.seterr(divide="ignore")
       try:
              identifier = sys.argv[1]
       except:
              identifier = time.strftime("%d%m%H")
       try:
              runnum = sys.argv[2]
       except:
              runnum = 1

       try:
              nevents = int(sys.argv[3])
       except:
              nevents = 10

       try:
              truelogmass = float(sys.argv[4])
       except:
              truelogmass = 0

       try:
              truelambdaval = float(sys.argv[5])
       except:
              truelambdaval = 0.5
       try:
              margcores = int(sys.argv[6])
       except:
              margcores = 10
       

       
       
       warnings.filterwarnings('ignore',category=UserWarning)
       sigdistsetup = makedist

       
       
       # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
       np.seterr(divide='ignore', invalid='ignore')

       try:
              os.mkdir('data')
       except:
              print("data folder already exists (good)")
       try:
              os.mkdir(f'data/{identifier}')
       except:
              print("Stem Folder Already Exists (good)")
       try:
              os.mkdir(f'data/{identifier}/{runnum}')
       except:
              raise Exception(f"The folder data/{identifier}/{runnum} already exists, stopping computation so files are not accidentally overwritten.")



       sigdist = sigdistsetup(truelogmass)


       nsig = int(np.round(truelambdaval*nevents))
       nbkg = int(np.round((1-truelambdaval)*nevents))
       sigsamples = log10eaxis[inverse_transform_sampling(sigdist(axis)+logjacob,nsig)]


       sigsamples_measured = []
       for sigsample in tqdm(sigsamples, desc="Creating measured signal vals", ncols=100):
              sigsamples_measured.append(log10eaxis[inverse_transform_sampling(edisp(axis,sigsample)+logjacob,Nsamples=1)])
       sigsamples_measured = np.array(sigsamples_measured)


       bkgsamples = log10eaxis[inverse_transform_sampling(bkgdist(log10eaxis)+logjacob,nbkg)]

       bkgsamples_measured = []
       for bkgsample in tqdm(bkgsamples, desc="Creating measured background vals", ncols=100):
              bkgsamples_measured.append(log10eaxis[inverse_transform_sampling(edisp(log10eaxis,bkgsample)+logjacob,Nsamples=1)])
       bkgsamples_measured = np.array(bkgsamples_measured)


       np.save(f"data/{identifier}/{runnum}/truesigsamples.npy", sigsamples)
       np.save(f"data/{identifier}/{runnum}/meassigsamples.npy", sigsamples_measured)
       np.save(f"data/{identifier}/{runnum}/truebkgsamples.npy", bkgsamples)
       np.save(f"data/{identifier}/{runnum}/measbkgsamples.npy", bkgsamples_measured)
       np.save(f"data/{identifier}/{runnum}/params.npy",         np.array([['lambda', 'Nsamples', 'logmass'],
                                          [truelambdaval, nevents, truelogmass]]))

       print("Done simulation.")
       truevals             = np.concatenate((sigsamples, bkgsamples))
       measuredvals         = np.concatenate((sigsamples_measured,bkgsamples_measured))

       

       edispnorms = np.array([special.logsumexp(edisp(log10eaxis,axisval)+logjacob) for axisval in log10eaxis])

       if -np.inf in edispnorms:
              print(COLOR.BOLD+"Your energy dispersion normalisation has -np.inf inside, which will almostly definitely mean your energy dispersion or the normalisation is wrong."+COLOR.END)

       edisplist = []


       print(f"There are {nevents} events being analyzed.")
       for i, sample in tqdm(enumerate(measuredvals),desc="Calculating edisp vals", ncols=100):
              edisplist.append(edisp(sample,log10eaxis)-edispnorms)
       edisplist = np.array(edisplist)

       bkgmargresults = []
       indices = np.arange(len(list(measuredvals)))
       with Pool(margcores) as pool:
              bkgfunc = functools.partial(marg, edisplist=edisplist, dist=bkgdist,log10eaxis=log10eaxis, print_progress=False)

              for result in tqdm(pool.imap(bkgfunc, indices), ncols=100, desc="Calculating background marginalisations"):
                     bkgmargresults.append(result)
              
              pool.close()
       print("Done calculating the background marginalisations")
       
       

       propmargresults = []
       indices = np.arange(len(list(measuredvals)))
       with Pool(margcores) as pool:
              propfunc = functools.partial(marg, edisplist=edisplist, dist=logpropdist,log10eaxis=log10eaxis, print_progress=False)

              for result in tqdm(pool.imap(propfunc, indices), ncols=100, desc="Calculating proposal marginalisations"):
                     propmargresults.append(result)
              
              pool.close()
       print("Done calculating the proposal marginalisations.")

       np.save(f'data/{identifier}/{runnum}/edisplist.npy', edisplist)
       np.save(f'data/{identifier}/{runnum}/bkgmargresults.npy', bkgmargresults)
       np.save(f'data/{identifier}/{runnum}/propmargresults.npy', propmargresults)


       # chime.info('sonic')







