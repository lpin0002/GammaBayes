from scipy import integrate, special, interpolate, stats, linalg
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import dynesty

axis = np.linspace(-2,2,100)

sampledist = lambda x: stats.uniform(loc=-2., scale=4).logpdf(x)

energydisp = lambda x: stats.norm(loc=0.0, scale=0.5).logpdf(x)



# Define the dimensionality of our problem.
ndim = 1


def loglike(x):
    return float(stats.norm(loc=0.0, scale=0.5).logpdf(x))

# Define our uniform prior via the prior transform.
def ptform(u):
    return 4. * u - 2.

# Sample from our distribution.
sampler = dynesty.NestedSampler(loglike, ptform, ndim,
                                bound='single', nlive=1000)
sampler.run_nested(dlogz=0.1)
res = sampler.results

marglist = np.exp(energydisp(axis)+sampledist(axis))
marglistint = integrate.simps(y=marglist, x=axis)
plt.figure()
histvals = plt.hist(res["samples"], bins=axis.shape[0])
plt.plot(axis, marglist/np.max(marglist)*np.max(histvals[0]))
plt.show()

print(np.exp(res['logz'][-1]), marglistint)





energydisp = lambda x: stats.norm(loc=1.0, scale=0.5).logpdf(x)



def loglike(x):
    return float(energydisp(x))


# Sample from our distribution.
sampler = dynesty.NestedSampler(loglike, ptform, ndim,
                                bound='single', nlive=1000)
sampler.run_nested(dlogz=0.1)
res = sampler.results

marglist = np.exp(energydisp(axis)+sampledist(axis))
marglistint = integrate.simps(y=marglist, x=axis)
plt.figure()
histvals = plt.hist(res["samples"], bins=axis.shape[0])
plt.plot(axis, marglist/np.max(marglist)*np.max(histvals[0]))
plt.show()

print(np.exp(res['logz'][-1]), marglistint)