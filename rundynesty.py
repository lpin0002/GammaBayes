
from scipy import integrate, interpolate, special
import numpy as np
import dynesty, warnings
import matplotlib.pyplot as plt


def rundynesty(logmeasuredval , logprior, logedisplist, axis, nlive = 4000):
    eaxis = 10**axis
    logjacob = np.log(eaxis)+np.log(axis[1]-axis[0])+np.log(np.log(10))
    def makeloglike(loglikearray=logedisplist):
        loglike = interpolate.interp1d(x=axis, y=loglikearray, bounds_error=False, fill_value=(loglikearray[0], loglikearray[-1]))
        def gaussfull(evalue):
            logevalue = np.log10(evalue)
            output = loglike(logevalue)
            
            # norm = 0
            if output.shape==(1,):
                return output[0]
            else:
                return output
        return gaussfull
    
    

    # Define our uniform prior via the prior transform.
    def makeptform(logpriorfunc=logprior):
        logpriorarray = logpriorfunc(axis)+logjacob
        logpriorarray = logpriorarray-special.logsumexp(logpriorarray)
        logcdfarray = np.logaddexp.accumulate(logpriorarray)
        cdfarray = np.exp(logcdfarray-logcdfarray[-1])
        # print(cdfarray[-1])
        interpfunc = interpolate.interp1d(y=np.arange(0,len(cdfarray)), x=cdfarray, bounds_error=False, fill_value=(0,len(cdfarray)-1), kind='nearest')
        
        def ptform(u):
            index = int(interpfunc(u[0]))
            u[0] = eaxis[index]
            return u
        return ptform

    # Sample from our distribution.
    ndim = 1
    warnings.filterwarnings('ignore',category=UserWarning)
    sampler = dynesty.NestedSampler(makeloglike(loglikearray=logedisplist), makeptform(logprior), ndim=ndim,
                                    bound='single', nlive=nlive)
    sampler.run_nested(dlogz=0.01, print_progress=False)#, print_func = custom_print_function)
    res = sampler.results

    return res