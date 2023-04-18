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
#Constant ratio offset between direct and nested no jacob= 0.30989914742863683
# firstratio = marglistint/np.exp(res['logz'][-1]) = 0.30989914742863683
# with jacob = 0.008514387720831325

axis = np.linspace(-2,2,1000)
binwidth = axis[1]-axis[0]

logbinwidth = np.log(binwidth)
print("binwidthstuff: ", binwidth, logbinwidth)


nlive = 1024
dlogz = 0.2
likelihoodspread = 0.5
priorspread = 4
priorcentre = -2
ndim = 1
measuredval = 0.6
measuredval2 = 0.3
eaxis_mod = 0.


def energydisp(erecon, etrue):
    func = stats.norm(loc=etrue, scale=likelihoodspread)
    val = func.logpdf(erecon)- np.log(integrate.simps(y=func.pdf(axis),x=axis))
    return val



def makedist(centre=priorcentre, spread =priorspread):
    func = stats.uniform(loc=centre, scale=spread)

    sampledist = lambda x: func.logpdf(x) - special.logsumexp(func.logpdf(axis)+logbinwidth)
    return sampledist 

def makeloglike(measured=measuredval):
    def loglike(x):
        return float(energydisp(measured, x))
    return loglike

# Define our uniform prior via the prior transform.

def makeptform(centre=priorcentre):
    sampledist = makedist(priorcentre)
    # def ptform(u):
    #     logpmf = sampledist(axis)
    #     logpmf = logpmf - special.logsumexp(logpmf)
    #     pmf = np.exp(logpmf)
    #     cdf = np.cumsum(pmf)
    #     try:
    #         index = np.searchsorted(cdf, u[0])
    #         u[0] = axis[index]
    #     except:
    #         if index>=axis.shape[0]-1:
    #             u[0] = axis[-1]
    #         else:
    #             raise Exception("Something is wrong with the uniform sampling cube in tester.py.")
        
    #     return u

    def ptform(u):
        u[0] = 4*u[0]-2
        return u
    return ptform


# Sample from our distribution.
sampler = dynesty.NestedSampler(makeloglike(measuredval), makeptform(priorcentre), ndim,
                                bound='single', nlive=nlive, logl_args=None)
sampler.run_nested(dlogz=dlogz)
res = sampler.results

marglist = energydisp(measuredval, axis) + makedist(priorcentre)(axis) + logbinwidth
marglistint = np.exp(special.logsumexp(marglist))
plt.figure()
histvals = plt.hist(res["samples"], bins=int(axis.shape[0]/5))
plt.plot(axis, np.exp(marglist)/np.max(np.exp(marglist))*np.max(histvals[0]))
plt.savefig("NestedSampling_vs_DirectIntegration.png")
plt.show()

print(np.exp(res['logz'][-1]), marglistint, marglistint/np.exp(res['logz'][-1]))
lnz_truth = np.log(marglistint)# analytic evidence solution
fig, axes = dyplot.runplot(res, lnz_truth=lnz_truth)
plt.savefig('ConvergencePlot.png')
plt.show()
plt.close()
print('\n\n')




# Sample from our distribution.
sampler = dynesty.NestedSampler(makeloglike(measuredval2), makeptform(priorcentre), ndim,
                                bound='single', nlive=nlive)
sampler.run_nested(dlogz=dlogz)
res = sampler.results

marglist = energydisp(measuredval2, axis) + makedist(priorcentre)(axis) + logbinwidth
marglistint = np.exp(special.logsumexp(marglist))
plt.figure()
histvals = plt.hist(res["samples"], bins=int(axis.shape[0]/5))
plt.plot(axis, np.exp(marglist)/np.max(np.exp(marglist))*np.max(histvals[0]))
plt.savefig("NestedSampling_vs_DirectIntegration.png")
plt.show()

print(np.exp(res['logz'][-1]), marglistint, marglistint/np.exp(res['logz'][-1]))
lnz_truth = np.log(marglistint)# analytic evidence solution
fig, axes = dyplot.runplot(res, lnz_truth=lnz_truth)
plt.savefig('ConvergencePlot.png')
plt.show()
print('\n\n')