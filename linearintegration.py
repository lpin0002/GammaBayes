import numpy as np
import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
from scipy import linalg, stats, integrate, special, interpolate
import chime
from utils import axis
from rundynesty import rundynesty

log10eaxis = axis
eaxis = 10**log10eaxis

measured_log10e = -1

logjacob = np.log(eaxis)+np.log(np.log(10))

priorbase   = stats.uniform(loc=10**log10eaxis[0], scale=10**log10eaxis[-1]-10**log10eaxis[0])
logprior    = priorbase.logpdf

likelihoodbase = stats.norm(loc=10**measured_log10e, scale=0.3*10**measured_log10e)
loglikelihood = likelihoodbase.logpdf

logedisplist = loglikelihood(eaxis)
##### DyNesty Stuff


import numpy as np



def makeloglike(loglikearray=logedisplist):
    loglike = interpolate.interp1d(x=log10eaxis, y=loglikearray, bounds_error=False, fill_value=(loglikearray[0], loglikearray[-1]))
    def gaussfull(cube):
        logevalue = cube[0]
        output = loglike(logevalue)
        return output
    return gaussfull



def makeptform(logpriorfunc=logprior):
    logpriorarray = logpriorfunc(eaxis)+logjacob
    logpriorarray = logpriorarray-special.logsumexp(logpriorarray)
    logcdfarray = np.logaddexp.accumulate(logpriorarray)
    cdfarray = np.exp(logcdfarray-logcdfarray[-1])
    # print(cdfarray[-1])
    interpfunc = interpolate.interp1d(y=np.arange(0,len(cdfarray)), x=cdfarray, bounds_error=False, fill_value=(0,len(cdfarray)-1), kind='nearest')
    
    def ptform(u):
        index = int(interpfunc(u[0]))
        u[0] = log10eaxis[index]
        return u
    return ptform



sampler = dynesty.NestedSampler(makeloglike(loglikearray=logedisplist), makeptform(logpriorfunc=logprior), ndim=1, nlive=5000, bound='multi', update_interval=1.0)
sampler.run_nested(dlogz=0.1)
res = sampler.results



################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################


# def rundynesty(logprior, logedisplist, log10eaxis, nlive = 5000, print_progress=False):
#     eaxis = 10**log10eaxis
#     logjacob = np.log(eaxis)+np.log(np.log(10))
#     def makeloglike(loglikearray=logedisplist):
#         loglike = interpolate.interp1d(x=log10eaxis, y=loglikearray, bounds_error=False, fill_value=(loglikearray[0], loglikearray[-1]))
#         def gaussfull(cube):
#             logevalue = cube[0]
#             output = loglike(logevalue)
#             return output
#         return gaussfull



#     def makeptform(logpriorfunc=logprior):
#         logpriorarray = logpriorfunc(eaxis)+logjacob
#         logpriorarray = logpriorarray-special.logsumexp(logpriorarray)
#         logcdfarray = np.logaddexp.accumulate(logpriorarray)
#         cdfarray = np.exp(logcdfarray-logcdfarray[-1])
#         # print(cdfarray[-1])
#         interpfunc = interpolate.interp1d(y=np.arange(0,len(cdfarray)), x=cdfarray, bounds_error=False, fill_value=(0,len(cdfarray)-1), kind='nearest')
        
#         def ptform(u):
#             index = int(interpfunc(u[0]))
#             u[0] = log10eaxis[index]
#             return u
#         return ptform



#     sampler = dynesty.NestedSampler(makeloglike(loglikearray=logedisplist), makeptform(logpriorfunc=logprior), ndim=1, 
#                                     nlive=nlive, bound='multi', update_interval=1.0)
#     sampler.run_nested(dlogz=0.1, print_progress=print_progress)
#     res = sampler.results

#     # To get equally weighted samples like MCMC use res.samples_equal()

#     return res

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################





againres = rundynesty(logprior, logedisplist, log10eaxis, print_progress=True)

overthetopeaxis = 10**np.linspace(log10eaxis[0],log10eaxis[-1], 100000000)

nearexactintegral = np.log(integrate.simps(y=np.exp(logprior(overthetopeaxis)+loglikelihood(overthetopeaxis)),x=overthetopeaxis))
print(nearexactintegral)


# Sound telling me the main computations are done
chime.success()



# dE = E ln(10) d(log_10(E))
postvals = np.array(loglikelihood(eaxis)+logprior(eaxis)+logjacob)
plt.figure()
histvals = plt.hist(res.samples_equal(), bins=log10eaxis-0.001*(log10eaxis[1]-log10eaxis[0]),alpha=0.6)
againhistvals = plt.hist(againres.samples_equal(), bins=log10eaxis-0.001*(log10eaxis[1]-log10eaxis[0]),alpha=0.6)
plt.plot(log10eaxis, np.exp(postvals)/np.max(np.exp(postvals))*np.max(histvals[0]), c='ForestGreen')
plt.axvline(measured_log10e, lw=0.5, c='r')
plt.savefig('Figures/LatestFigures/NestedSampling_vs_Direct.png')
plt.show()



print(f'\nlog direct = {nearexactintegral},log nested = {res.logz[-1]}, ratio=log nested/log direct={res.logz[-1]/nearexactintegral}')
# Plotting the convergence plot
fig = plt.subplots(4,1, figsize=(14,7))
fig, axes = dyplot.runplot(res, lnz_truth=nearexactintegral, fig=fig)
plt.savefig("Figures/LatestFigures/ConvergencePlot.pdf")
plt.show()
print('\n')



print(f'\n (Again) log direct = {nearexactintegral},log nested = {againres.logz[-1]}, ratio=log nested/log direct={againres.logz[-1]/nearexactintegral}')
# Plotting the convergence plot
fig = plt.subplots(4,1, figsize=(14,7))
fig, axes = dyplot.runplot(againres, lnz_truth=nearexactintegral, fig=fig)
plt.savefig("Figures/LatestFigures/ConvergencePlot.pdf")
plt.show()
print('\n')