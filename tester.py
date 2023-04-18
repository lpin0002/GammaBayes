from scipy import integrate, special, interpolate, stats, linalg
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import dynesty
from dynesty import plotting as dyplot
import math
from utils import axis, eaxis_mod
print('\n')



nlive = 1024
dlogz = 0.1
likelihoodspread = 0.5
priorspread = 0.3
priorcentre = 0.5
ndim = 1
measuredval = 0.0



energydisp = lambda erecon, etrue: stats.norm(loc=etrue, scale=likelihoodspread).logpdf(erecon)

def makedist(centre=priorcentre, spread =priorspread):
    sampledist = lambda x: stats.norm(loc=centre, scale=spread).logpdf(x)
    return sampledist 

def makeloglike(measured=measuredval):
    def loglike(x):
        return float(energydisp(measured, x)+np.log(np.power(10.,x)))
    return loglike

# Define our uniform prior via the prior transform.

def makeptform(centre=priorcentre):
    sampledist = makedist(priorcentre)
    def ptform(u):
        logpmf = sampledist(axis)
        logpmf = logpmf - special.logsumexp(logpmf)
        pmf = np.exp(logpmf)
        cdf = np.cumsum(pmf)
        index = np.searchsorted(cdf, u[0])
        u[0] = axis[index]
        return u
    return ptform

endsample = 1
# Sample from our distribution.
sampler = dynesty.NestedSampler(makeloglike(measuredval), makeptform(priorcentre), ndim,
                                bound='single', nlive=nlive)
sampler.run_nested(dlogz=dlogz)
res = sampler.results

marglist = energydisp(measuredval, axis)+makedist(priorcentre)(axis)+eaxis_mod
marglistint = np.exp(special.logsumexp(marglist))*(axis[1]-axis[0])
plt.figure()
histvals = plt.hist(res["samples"][:-endsample], bins=int(axis.shape[0]/5))
plt.plot(axis, np.exp(marglist)/np.max(np.exp(marglist))*np.max(histvals[0]))
plt.show()

print(np.exp(res['logz'][-1]), marglistint)






# Sample from our distribution.
sampler = dynesty.NestedSampler(makeloglike(measuredval), makeptform(priorcentre), ndim,
                                bound='single', nlive=nlive)
sampler.run_nested(dlogz=dlogz)
res = sampler.results

marglist = energydisp(measuredval, axis)+makedist(priorcentre)(axis)+eaxis_mod
marglistint = np.exp(special.logsumexp(marglist))*(axis[1]-axis[0])
plt.figure()
histvals = plt.hist(res["samples"][:-endsample], bins=int(axis.shape[0]/5))
plt.plot(axis, np.exp(marglist)/np.max(np.exp(marglist))*np.max(histvals[0]))
plt.savefig("NestedSampling_vs_DirectIntegration.png")
plt.show()

print(np.exp(res['logz'][-1]), marglistint)
lnz_truth = np.log(marglistint)# analytic evidence solution
fig, axes = dyplot.runplot(res, lnz_truth=lnz_truth)
plt.savefig('ConvergencePlot.png')
plt.show()
print('\n\n')