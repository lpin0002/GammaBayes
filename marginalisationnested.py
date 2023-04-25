
from utils import inverse_transform_sampling, axis, bkgdist, makedist, edisp, eaxis_mod, COLOR,logjacob
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



def sigmarg(logmass, edisplist, sigdistsetup, measuredvals, logjacob=logjacob, log10eaxis=log10eaxis, print_progress=False):
       tempsigdist = sigdistsetup(logmass)
       tempmarglogmassrow = []
       tempmarglogmassrowsamplers = []
       tempsigdist = sigdistsetup(logmass)
       for i, sample in enumerate(measuredvals):
              tempsigmargfullvalarray = rundynesty(tempsigdist, edisplist[i], log10eaxis=log10eaxis, print_progress=print_progress)
              tempmarglogmassrowsamplers.append(tempsigmargfullvalarray)
              tempsigmarg = tempsigmargfullvalarray.logz[-1]
              tempmarglogmassrow.append(tempsigmarg)
       return np.array(tempmarglogmassrow), tempmarglogmassrowsamplers


def sigmargwrapper(logmass, edisplist, sigdistsetup, measuredvals, print_progress):
       return sigmarg(logmass=logmass, edisplist=edisplist, sigdistsetup=sigdistsetup, measuredvals=measuredvals, print_progress=print_progress)


def bkgmarg(index, edisplist, bkgdist, log10eaxis=log10eaxis, print_progress=False):
       bkgmargsampler = rundynesty(bkgdist, edisplist[index], log10eaxis=log10eaxis, print_progress=print_progress)
       return bkgmargsampler.logz[-1]


def bkgmargwrapper(index, edisplist, bkgdist, log10eaxis, print_progress):
       return bkgmarg(index, edisplist, bkgdist, log10eaxis, print_progress)


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
       try:
              nbinslogmass = int(sys.argv[7])
       except:
              nbinslogmass = 21
       try:
              nbinslambda = int(sys.argv[8])
       except:
              nbinslambda = 21
       try:
              totalrunevents = int(sys.argv[9])
       except:
              totalrunevents = nevents
       

       
       
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

       logmasswindowwidth   = 2/np.sqrt(totalrunevents)
       logmasslowerbound    = truelogmass-logmasswindowwidth
       logmassupperbound    = truelogmass+logmasswindowwidth

       lambdavalwindowwidth = 5/np.sqrt(totalrunevents)
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
       np.save(f'data/{identifier}/{runnum}/logmassrange_nested.npy',logmassrange)
       np.save(f'data/{identifier}/{runnum}/lambdarange_nested.npy',lambdarange)
       # lambdarange = np.array([0.45, 0.5])
       print("logmassrange: ", logmassrange[0], logmassrange[-1])
       print("lambdarange: ", lambdarange[0], lambdarange[-1])

       

       edispnorms = np.array([special.logsumexp(edisp(log10eaxis,axisval)+logjacob) for axisval in log10eaxis])

       if -np.inf in edispnorms:
              print(COLOR.BOLD+"Your energy dispersion normalisation has -np.inf inside, which will almostly definitely mean your energy dispersion or the normalisation is wrong."+COLOR.END)

       edisplist = []
       # bkgmarglist = []
       bkgdistnormed = bkgdist(log10eaxis) - special.logsumexp(bkgdist(log10eaxis)+logjacob)


       print(f"There are {nevents} events being analyzed.")
       for i, sample in tqdm(enumerate(measuredvals),desc="Calculating edisp vals", ncols=100):
              edisplist.append(edisp(sample,log10eaxis)-edispnorms)
              # bkgmarglist.append(special.logsumexp(bkgdistnormed+edisplist[i]+logjacob))
       edisplist = np.array(edisplist)

       bkgmarglist = []
       indices = np.arange(len(list(measuredvals)))
       with Pool(margcores) as pool:
              bkgfunc = functools.partial(bkgmarg, edisplist=edisplist, bkgdist=bkgdist,log10eaxis=log10eaxis, print_progress=False)

              for result in tqdm(pool.imap(bkgfunc, indices), ncols=100, desc="Calculating background marginalisations..."):
                     bkgmarglist.append(result)
              
              pool.close()
       print("Done calculating the background marginalisations")
       
       np.save(f'data/{identifier}/{runnum}/edisplist_nested.npy', edisplist)
       np.save(f'data/{identifier}/{runnum}/bkgmarglist_nested.npy', bkgmarglist)

       # sigmargsamplers = []
       sigmarglogzvals = []
       # num_cores = multiprocessing.cpu_count()
       print(f"You have allocated {margcores} cores on your machine")
       with Pool(margcores) as pool:
              func = functools.partial(sigmargwrapper, edisplist=edisplist, sigdistsetup=sigdistsetup, measuredvals=measuredvals, print_progress=False)
       
              for result in tqdm(pool.imap(func, logmassrange), total=len(list(logmassrange)), ncols=100, desc="Calculating signal marginalisations..."):
                     evidencevals, samplers = result
                     sigmarglogzvals.append(evidencevals)
                     # sigmargsamplers.append(samplers)

              pool.close()
       print("Done calculating the signal marginalisations.")

       np.save(f'data/{identifier}/{runnum}/sigmarglogzvals_nested.npy', sigmarglogzvals)
       # np.save(f'data/{identifier}/{runnum}/sigmargsamplers_nested.npy', sigmargsamplers)


       # chime.info('sonic')







