from scipy import integrate, special, interpolate, stats
import os, sys, time, random, chime, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm
from utils import inverse_transform_sampling, axis, makedist, edisp, bkgdist, eaxis_mod, eaxis, logjacob
from BFCalc.BFInterp import DM_spectrum_setup
# Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
np.seterr(divide='ignore', invalid='ignore')
# Check that jacobian for this script is correct
try:
    identifier = sys.argv[1]
except:
    identifier = time.strftime("%d%m%H%M")
sigdistsetup = makedist

truelogmass = 1.3

sigdist = sigdistsetup(truelogmass)


nevents = 100
lambdaval = 0.5
nsig = int(np.round(lambdaval*nevents))
nbkg = int(np.round((1-lambdaval)*nevents))
sigsamples = axis[inverse_transform_sampling(sigdist(axis)+logjacob,nsig)]

sigsamples_measured = []
for sigsample in tqdm(sigsamples, desc="Creating measured signal vals", ncols=100):
    sigsamples_measured.append(axis[inverse_transform_sampling(edisp(axis,sigsample)+logjacob,Nsamples=1)])
sigsamples_measured = np.array(sigsamples_measured)


bkgsamples = axis[inverse_transform_sampling(bkgdist(axis)+logjacob,nbkg)]

bkgsamples_measured = []
for bkgsample in tqdm(bkgsamples, desc="Creating measured background vals", ncols=100):
    bkgsamples_measured.append(axis[inverse_transform_sampling(edisp(axis,bkgsample)+logjacob,Nsamples=1)])
bkgsamples_measured = np.array(bkgsamples_measured)


backgroundintegrals = []
signalintegrals = []
for i in range(len(axis[1:])):
    evals = np.linspace(10**axis[i],10**axis[i+1],100)
    signalintegrals.append(np.exp(special.logsumexp(sigdist(np.log10(evals))+np.log(evals))))
    backgroundintegrals.append(np.exp(special.logsumexp(bkgdist(np.log10(evals))+np.log(evals))))
signalintegrals = np.array(signalintegrals)
signalintegrals = np.array(signalintegrals)


centrevals = axis[:-1]+0.5*(axis[1]-axis[0])

chime.info('sonic')

plt.figure()
plt.title("signal true values")
sighistvals = plt.hist(sigsamples, bins=centrevals, alpha=0.7, label='Measured signal')
sigdistvals = np.exp(sigdist(axis))*eaxis
plt.plot(axis, sigdistvals/np.max(sigdistvals)*np.max(sighistvals[0]), label='point signal with jacobian')
plt.plot(centrevals, signalintegrals/np.max(signalintegrals)*np.max(sighistvals[0]), label='signal integral vals')
plt.legend()
plt.savefig("Figures/LatestFigures/TrueValsSignal.pdf")
plt.show()



plt.figure()
plt.title("background true values")
bkghistvals = plt.hist(bkgsamples, bins=centrevals, alpha=0.7, label="Measured background")
bkgdistvals = np.exp(bkgdist(axis))*eaxis
plt.plot(axis, bkgdistvals/np.max(bkgdistvals)*np.max(bkghistvals[0]), label='point background with jacobian')
plt.plot(centrevals, backgroundintegrals/np.max(backgroundintegrals)*np.max(bkghistvals[0]), label='background integral vals')

plt.legend()
plt.savefig("Figures/LatestFigures/TrueValsBackground.pdf")
plt.show()


plt.figure()
plt.title("measured values")
plt.hist(sigsamples_measured, bins=centrevals, alpha=0.7, label='pseudo-measured signal')
plt.hist(bkgsamples_measured, bins=centrevals, alpha=0.7, label='pseudo-measured background')
plt.legend()
plt.savefig("Figures/LatestFigures/MeasuredVals.pdf")
plt.show()


np.save("data/truesigsamples.npy", sigsamples)
np.save("data/meassigsamples.npy", sigsamples_measured)
np.save("data/truebkgsamples.npy", bkgsamples)
np.save("data/measbkgsamples.npy", bkgsamples_measured)
np.save("data/params.npy",         np.array([['lambda', 'Nsamples', 'logmass'],
                                        [lambdaval, nevents, truelogmass]]))