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

logmasslowerbound = truelogmass-20/np.sqrt(nevents)
if logmasslowerbound<axis[0]:
       logmasslowerbound = axis[0]
logmassrange = np.linspace(logmasslowerbound,truelogmass+20/np.sqrt(nevents),61)
# lambdarange = np.linspace(truelambdaval-2/np.sqrt(nevents),truelambdaval+2/np.sqrt(nevents),41)
lambdarange = np.array([0.45, 0.5])
normedlogposterior = np.load("normedlogposterior.npy")



plt.figure()
plt.pcolormesh(lambdarange, logmassrange, np.exp(normedlogposterior))
plt.ylabel("log mass [TeV]")
plt.xlabel("lambda = signal events/total events")
plt.colorbar(label="Probability Density [1/TeV]")
plt.axvline(truelogmass, c='r')
plt.axhline(truelambdaval, c='r')
plt.savefig("posterior.pdf")
plt.show()

plt.figure()
plt.axvline(logmassrange[np.abs(logmassrange-truelogmass).argmin()+3])
plt.axvline(logmassrange[np.abs(logmassrange-truelogmass).argmin()+2])
plt.axvline(logmassrange[np.abs(logmassrange-truelogmass).argmin()+1])
plt.axvline(logmassrange[np.abs(logmassrange-truelogmass).argmin()])
plt.axvline(logmassrange[np.abs(logmassrange-truelogmass).argmin()-1])
plt.axvline(logmassrange[np.abs(logmassrange-truelogmass).argmin()-2])
plt.axvline(logmassrange[np.abs(logmassrange-truelogmass).argmin()-3])
plt.plot(logmassrange, np.exp(normedlogposterior[:, np.abs(truelambdaval-lambdarange).argmin()]))
plt.axvline(truelogmass, c='r', label=params[1,2])
plt.xlabel("log mass [TeV]")
plt.ylabel("Probability density (slice) [1/TeV]")
plt.legend()
plt.savefig("logmassslice.pdf")
plt.show()


plt.figure()
plt.plot(lambdarange, np.exp(normedlogposterior[:, np.abs(truelogmass-logmassrange).argmin()]))
plt.xlabel("lambda = signal events/total events")
plt.ylabel("Probability density (slice) []")
plt.axvline(truelambdaval,c='r', label=params[1,0])
plt.legend()
plt.savefig("lambdaslice.pdf")
plt.show()