
from utils import inverse_transform_sampling, axis, bkgdist, makedist, edisp

from scipy import integrate, special, interpolate, stats
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

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


logmassrange = np.linspace(-2,2,81)

edispnorms = np.array([special.logsumexp(edisp(axis,axisval)) for axisval in axis])

edisplist = []
for sample in measuredvals:
        edisplist.append(edisp(sample,axis)-edispnorms)
edisplist = np.array(edisplist)

logmassposterior = []
bkgdistnormed = bkgdist(axis) - special.logsumexp(bkgdist(axis))
for logmass in tqdm(logmassrange):
        tempsigdist = makedist(logmass)
        tempsigdistaxis = tempsigdist(axis) - special.logsumexp(tempsigdist(axis))

        tempmargval = 0
        for i, sample in enumerate(measuredvals):
                tempval = np.logaddexp(np.log(0.5)+tempsigdistaxis,np.log(0.5)+bkgdistnormed)
                # print(tempval)
                tempmargval += special.logsumexp(tempval+edisplist[i])
        
        logmassposterior.append(tempmargval)

plt.figure()
plt.plot(logmassrange, np.exp(logmassposterior - special.logsumexp(logmassposterior)))
plt.axvline(1.0)
plt.show()


