
from utils import inverse_transform_sampling, axis, bkgdist, makedist, edisp, eaxis_mod
from scipy import integrate, special, interpolate, stats
import numpy as np
import os, time, random
from tqdm import tqdm
import matplotlib.pyplot as plt

sigsamples          = np.load("truesigsamples.npy")
sigsamples_measured = np.load("meassigsamples.npy")
bkgsamples          = np.load("truebkgsamples.npy")
bkgsamples_measured = np.load("measbkgsamples.npy")
truevals            = np.concatenate((sigsamples, bkgsamples))
measuredvals        = np.concatenate((sigsamples_measured,bkgsamples_measured))

plt.figure()
plt.title("measured values")
centrevals = axis[:-1]+0.5*(axis[1]-axis[0])
plt.hist(sigsamples_measured, bins=centrevals, alpha=0.7)
plt.hist(bkgsamples_measured, bins=centrevals, alpha=0.7)
plt.show()


logmassrange = np.linspace(0.495,0.505,41)
lambdarange = np.array([0.49, 0.50])

edispnorms = np.array([special.logsumexp(edisp(axis,axisval)+eaxis_mod) for axisval in axis])

edisplist = []
for sample in measuredvals:
        edisplist.append(edisp(sample,axis)-edispnorms)
edisplist = np.array(edisplist)

logmassposterior = []
bkgdistnormed = bkgdist(axis) - special.logsumexp(bkgdist(axis)+eaxis_mod)
for lambdaval in tqdm(lambdarange):
    templambdarow = []
    for logmass in logmassrange:
            tempsigdist = makedist(logmass)
            tempsigdistaxis = tempsigdist(axis) - special.logsumexp(tempsigdist(axis)+eaxis_mod)

            tempmargval = 0
            for i, sample in enumerate(measuredvals):
                    tempval = np.logaddexp(np.log(0.5)+tempsigdistaxis,np.log(0.5)+bkgdistnormed)
                    # print(tempval)
                    tempmargval += special.logsumexp(tempval+edisplist[i]+eaxis_mod)
            
            templambdarow.append(tempmargval)
    logmassposterior.append(templambdarow)


normedposterior = np.exp(logmassposterior - special.logsumexp(logmassposterior))
plt.figure()
plt.pcolormesh(logmassrange, lambdarange, normedposterior)
plt.axvline(1.0)
plt.axhline(0.5)
plt.savefig(time.strftime(f"posterior{measuredvals.shape[0]}%H%M.png"))
plt.savefig(f"posterior{measuredvals.shape[0]}.pdf")
plt.savefig("posterior.pdf")
plt.show()

plt.figure()
plt.plot(logmassrange, normedposterior[np.abs(0.5-lambdarange).argmin(),:])
plt.axvline(1.0)
plt.savefig(time.strftime(f"logmassslice{measuredvals.shape[0]}%H%M.png"))
plt.savefig(f"logmassslice{measuredvals.shape[0]}.pdf")
plt.savefig("logmassslice.pdf")
plt.show()
