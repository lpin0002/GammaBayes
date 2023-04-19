import numpy as np
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
from scipy import linalg, stats, integrate, special, interpolate
import chime
from utils import axis
measured = 0.1

eaxis = 10**axis

def makeloglike(measured):
    def gaussfull(value):
        func = stats.norm(loc=value,scale=5)
        output = func.logpdf(measured)
        
        # norm = 0
        if output.shape==(1,):
            norm = np.log(integrate.simps(y=func.pdf(eaxis), x=eaxis))
            return output[0]-norm
        else:
            norm = np.array([np.log(integrate.simps(y=stats.norm.pdf(eaxis,loc=val,scale=5), x=eaxis)) for val in value])
            return output-norm
    return gaussfull

prior = stats.uniform(loc=10**axis[0], scale=10**axis[1]-10**axis[0])

def priorfull(value):
    return prior.logpdf(value)-np.log(integrate.simps(y = prior.pdf(eaxis), x=eaxis))

logirf = lambda erecon, etrue: stats.norm(loc=etrue,scale=5).logpdf(erecon)

def rundynesty(prior=priorfull, loglike = logirf):
    def makeloglike(measured, loglike=loglike):
        def gaussfull(value):
            output = loglike(measured, value)
            
            # norm = 0
            if output.shape==(1,):
                norm = np.log(integrate.simps(y=np.exp(loglike(eaxis)), x=eaxis))
                return output[0]-norm
            else:
                norm = np.array([np.log(integrate.simps(y=stats.norm.pdf(eaxis,loc=val,scale=5), x=eaxis)) for val in value])
                return output-norm
        return gaussfull

    # Define our uniform prior via the prior transform.
    def makeptform(priorfunc):
        priorarray = np.exp(priorfunc(eaxis)+np.log(eaxis)+np.log(eaxis[1]-eaxis[0]))
        cdfarray = np.cumsum(priorarray/np.sum(priorarray))
        interpfunc = interpolate.interp1d(y=np.arange(0,len(cdfarray)), x=cdfarray, bounds_error=False, fill_value=(0,len(cdfarray)-1), kind='nearest')
        
        def ptform(u):
            index = int(interpfunc(u[0]))
            u[0] = eaxis[index]
            return u
        return ptform

    # Sample from our distribution.
    ndim = 1
    sampler = dynesty.NestedSampler(makeloglike(measured), makeptform(priorfull), ndim,
                                    bound='single', nlive=3000)
    sampler.run_nested(dlogz=0.1)
    res = sampler.results

    return res

res = rundynesty()
chime.info('sonic')

# dE = E ln(10) d(log_10(E))
postvals = np.exp(makeloglike(measured)(eaxis)+priorfull(eaxis))
plt.figure()
histvals = plt.hist(res.samples, bins=int(eaxis.shape[0]/10))
plt.plot(eaxis, postvals/np.max(postvals)*np.max(histvals[0]))
plt.axvline(measured)
plt.show()
# Plot results.
# lnz_truth = -np.log(2 * 10.)  # analytic evidence solution
lnz_direct = np.log(integrate.simps(y = postvals, x=eaxis))
# print(lnz_truth, lnz_direct, res.logz[-1])
print(f'log direct = {lnz_direct},log nested = {res.logz[-1]}, ratio=nested/direct={np.exp(res.logz[-1]-lnz_direct)}')
# fig = plt.subplots(4,1, figsize=(14,7))
# fig, axes = dyplot.runplot(res, lnz_truth=lnz_truth, fig=fig)
# plt.show()

fig = plt.subplots(4,1, figsize=(14,7))
fig, axes = dyplot.runplot(res, lnz_truth=lnz_direct, fig=fig)
plt.show()


