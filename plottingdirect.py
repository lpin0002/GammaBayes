from utils import inverse_transform_sampling, axis, bkgdist, makedist, edisp, eaxis_mod, color
from scipy import integrate, special, interpolate, stats
import numpy as np
import os, time, random
from tqdm import tqdm
import matplotlib.pyplot as plt
import chime
from BFCalc.BFInterp import DM_spectrum_setup
import warnings



sigsamples          = np.load("truesigsamples.npy")
sigsamples_measured = np.load("meassigsamples.npy")
bkgsamples          = np.load("truebkgsamples.npy")
bkgsamples_measured = np.load("measbkgsamples.npy")
params              = np.load("params.npy")
params[1,:]         = params[1,:]
truelogmass     = float(params[1,2])
nevents         = int(params[1,1])
truelambdaval   = float(params[1,0])
truevals            = np.concatenate((sigsamples, bkgsamples))
measuredvals        = np.concatenate((sigsamples_measured,bkgsamples_measured))

logmassrange = np.load('logmassrange.npy')
lambdarange = np.load('lambdarange.npy')

print(lambdarange)
normedlogposterior = np.load("normedlogposteriorDirect.npy")


plt.figure(dpi=300)
pcol = plt.pcolor(lambdarange, logmassrange, np.exp(normedlogposterior), snap=True)
pcol.set_edgecolor('face')
plt.ylabel("log mass [TeV]")
plt.xlabel("lambda = signal events/total events")
plt.colorbar(label="Probability Density [1/TeV]")
plt.axhline(truelogmass, c='r')
plt.axvline(truelambdaval, c='r')
plt.grid(False)
if measuredvals.shape[0]>10000:
       plt.title(str(measuredvals.shape[0]))
       plt.savefig(time.strftime(f"Figures/posterior%H%M_{measuredvals.shape[0]}Direct.pdf"))
plt.savefig("posteriorDirect.pdf")
plt.show()

plt.figure()
plt.plot(logmassrange, np.exp(normedlogposterior[:,np.abs(truelambdaval-lambdarange).argmin()]))
plt.axvline(truelogmass, c='r', label=params[1,2])
plt.xlabel("log mass [TeV]")
plt.ylabel("Probability density (slice) [1/TeV]")
plt.legend()
plt.savefig("logmasssliceDirect.pdf")
if measuredvals.shape[0]>10000:
       plt.title(str(measuredvals.shape[0]))
       plt.savefig(time.strftime(f"Figures/logmassslice%H%M_{measuredvals.shape[0]}Direct.pdf"))
plt.show()


plt.figure()
plt.plot(lambdarange, np.exp(normedlogposterior[np.abs(truelogmass-logmassrange).argmin(),:]))
plt.xlabel("lambda = signal events/total events")
plt.ylabel("Probability density (slice) []")
plt.axvline(truelambdaval,c='r', label=params[1,0])
plt.legend()
if measuredvals.shape[0]>10000:
       plt.title(str(measuredvals.shape[0]))
       plt.savefig(time.strftime(f"Figures/lambdaslice%H%M_{measuredvals.shape[0]}Direct.pdf"))
plt.savefig("lambdasliceDirect.pdf")
plt.show()