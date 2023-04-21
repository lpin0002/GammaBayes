
from scipy import integrate, interpolate, special
import numpy as np
import dynesty



def rundynesty(logmeasuredval , logprior, logedisp, axis, nlive = 500):
    eaxis = 10**axis
    logjacob = np.log(eaxis)+np.log(axis[1]-axis[0])+np.log(np.log(10))
    def makeloglike(measuredval=logmeasuredval, loglike=logedisp):
        def gaussfull(evalue):
            logevalue = np.log10(evalue)
            output = loglike(measuredval, logevalue)
            
            # norm = 0
            if output.shape==(1,):
                norm = special.logsumexp(loglike(axis, logevalue)+logjacob)
                return output[0]-norm
            else:
                norm = np.array([special.logsumexp(loglike(axis, logval)+logjacob) for logval in logevalue])
                return output-norm
        return gaussfull
    
    

    # Define our uniform prior via the prior transform.
    def makeptform(logpriorfunc=logprior):
        logpriorarray = logpriorfunc(axis)+logjacob
        logpriorarray = logpriorarray-special.logsumexp(logpriorarray)
        logcdfarray = np.logaddexp.accumulate(logpriorarray)
        cdfarray = np.exp(logcdfarray-logcdfarray[-1])
        # print(cdfarray[-1])
        print(cdfarray[-1])
        cdfarray = cdfarray/cdfarray[-1]
        interpfunc = interpolate.interp1d(y=np.arange(0,len(cdfarray)), x=cdfarray, bounds_error=False, fill_value=(0,len(cdfarray)-1), kind='nearest')
        
        def ptform(u):
            index = int(interpfunc(u[0]))
            u[0] = eaxis[index]
            return u
        return ptform

    # Sample from our distribution.
    ndim = 1
    sampler = dynesty.NestedSampler(makeloglike(logmeasuredval), makeptform(logprior), ndim=ndim,
                                    bound='single', nlive=nlive)
    sampler.run_nested(dlogz=0.1)#, print_func = custom_print_function)
    res = sampler.results

    return res