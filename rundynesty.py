
from scipy import integrate, interpolate, special
import numpy as np
import dynesty, warnings
# import matplotlib.pyplot as plt

def rundynesty(logprior, logedisplist, log10eaxis, nlive = 12000, print_progress=False):
    eaxis = 10**log10eaxis
    logjacob = np.log(eaxis)+np.log(np.log(10))
    def makeloglike(loglikearray=logedisplist):
        loglike = interpolate.interp1d(x=log10eaxis, y=loglikearray, bounds_error=False, fill_value=(loglikearray[0], loglikearray[-1]))
        def gaussfull(cube):
            logevalue = cube[0]
            output = loglike(logevalue)
            return output
        return gaussfull



    def makeptform(logpriorfunc=logprior):
        logpriorarray = logpriorfunc(log10eaxis)+logjacob
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


    sampler = dynesty.NestedSampler(makeloglike(loglikearray=logedisplist), makeptform(logpriorfunc=logprior), ndim=1, 
                                    nlive=nlive, bound='single')
    
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    sampler.run_nested(dlogz=0.05, print_progress=print_progress, maxcall=int(5e6))
    res = sampler.results
    warnings.filterwarnings("default")
    # To get equally weighted samples like MCMC use res.samples_equal()

    return res