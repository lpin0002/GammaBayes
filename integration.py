import numpy as np
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
from scipy import linalg, stats, integrate, special, interpolate
import chime
# from utils import axis
# Define the dimensionality of our problem.
# ratio=nested/direct=0.97658863284001
measured = 1.5
gauss= stats.norm(loc=10**measured,scale=0.1*10**measured)
eaxis = np.logspace(-2,2,200)

# eaxis = 10**axis

def gaussfull(value):
    return gauss.logpdf(value)

prior = stats.uniform(loc=eaxis[0],scale=eaxis[-1]-eaxis[0])

def priorfull(value):
    return prior.logpdf(value)

def loglike(x):
    return float(gaussfull(x))

# Define our uniform prior via the prior transform.

def makeptform(priorfunc):
    priorarray = np.exp(priorfull(eaxis)+np.log(eaxis))
    cdfarray = np.cumsum(priorarray/np.sum(priorarray))
    interpfunc = interpolate.interp1d(y=np.arange(0,len(cdfarray)), x=cdfarray, bounds_error=False, fill_value=(0,len(cdfarray)-1), kind='nearest')
    
    def ptform(u):
        # priorarray = np.exp(priorfull(eaxis))
        # cdfarray = np.cumsum(priorarray/np.sum(priorarray))
        # interpfunc = interpolate.interp1d(y=np.arange(0,len(cdfarray)), x=cdfarray, bounds_error=False, fill_value=(0,len(cdfarray)-1))
        # index = np.searchsorted(cdfarray, u[0], side='right')
        index = int(interpfunc(u[0]))
        u[0] = eaxis[index]
        # u[0]= (eaxis[-1]-eaxis[0])*u[0] + eaxis[0]
        return u
    return ptform

# Sample from our distribution.
ndim = 1
sampler = dynesty.NestedSampler(loglike, makeptform(priorfull), ndim,
                                bound='single', nlive=5000)
sampler.run_nested(dlogz=0.01)
res = sampler.results


chime.info('sonic')

# dE = E ln(10) d(log_10(E))
postvals = np.exp(gaussfull(eaxis)+priorfull(eaxis))
plt.figure()
histvals = plt.hist(res.samples, bins=int(eaxis.shape[0]/2))
plt.plot(eaxis, postvals/np.max(postvals)*np.max(histvals[0]))
plt.axvline(10**measured)
plt.show()
# Plot results.
# lnz_truth = -np.log(2 * 10.)  # analytic evidence solution
lnz_direct = np.log(integrate.simps(y = postvals, x=eaxis))
# print(lnz_truth, lnz_direct, res.logz[-1])
print(f'direct = {lnz_direct},nested = {res.logz[-1]}, ratio=nested/direct={res.logz[-1]/lnz_direct}')
# fig = plt.subplots(4,1, figsize=(14,7))
# fig, axes = dyplot.runplot(res, lnz_truth=lnz_truth, fig=fig)
# plt.show()

fig = plt.subplots(4,1, figsize=(14,7))
fig, axes = dyplot.runplot(res, lnz_truth=lnz_direct, fig=fig)
plt.show()


