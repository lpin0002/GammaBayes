from scipy import integrate, special, interpolate, stats
import os, sys, time, random, chime, numpy as np, matplotlib.pyplot as plt, warnings
from tqdm import tqdm
from utils import inverse_transform_sampling, log10eaxis, makedist, edisp, bkgdist, eaxis_mod, eaxis, logjacob
from BFCalc.BFInterp import DM_spectrum_setup
# Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
np.seterr(divide='ignore', invalid='ignore')
# Check that jacobian for this script is correct
try:
    identifier = sys.argv[1]
except:
    identifier = time.strftime("%d%m%H")

try:
    runnum = int(sys.argv[2])
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
    lambdaval = float(sys.argv[5])
except:
    lambdaval = 0.5

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

sigdistsetup = DM_spectrum_setup



sigdist = sigdistsetup(truelogmass)


nsig = int(np.round(lambdaval*nevents))
nbkg = int(np.round((1-lambdaval)*nevents))
sigsamples = log10eaxis[inverse_transform_sampling(sigdist(log10eaxis)+logjacob,nsig)]

sigsamples_measured = []
for sigsample in tqdm(sigsamples, desc="Creating measured signal vals", ncols=100):
    sigsamples_measured.append(log10eaxis[inverse_transform_sampling(edisp(log10eaxis,sigsample)+logjacob,Nsamples=1)])
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
                                        [lambdaval, nevents, truelogmass]]))

print("Done simulation.")