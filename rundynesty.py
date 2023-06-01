
from scipy import integrate, interpolate, special
import numpy as np
import dynesty, warnings
from utils import makelogjacob
# import matplotlib.pyplot as plt

def rundynesty(logprior, logedisplist, log10eaxis, nlive = 500, print_progress=False):
    logjacob = makelogjacob(log10eaxis)
    def makeloglike(loglikearray=logedisplist):
        loglikelihood = interpolate.interp1d(x=log10eaxis, y=loglikearray, bounds_error=False, fill_value=(-np.inf, -np.inf), kind='nearest')
        def gaussfull(cube):
            logevalue = cube[0]
            output = loglikelihood(logevalue)
            return output
        return gaussfull



    def makeptform(logpriorfunc=logprior):
        logpriorarray = logpriorfunc(log10eaxis)+logjacob
        logpriorarray = logpriorarray-special.logsumexp(logpriorarray)
        logcdfarray = np.logaddexp.accumulate(logpriorarray)
        cdfarray = np.exp(logcdfarray-logcdfarray[-1])
        # print(cdfarray[-1])
        interp_inverse_cdf = interpolate.interp1d(y=np.arange(0,len(cdfarray)), x=cdfarray, bounds_error=False, 
                                          fill_value=(0,len(cdfarray)-1), kind='nearest')
        
        def ptform(u):
            index = int(interp_inverse_cdf(u[0]))
            u[0] = log10eaxis[index]
            return u
        return ptform


    sampler = dynesty.NestedSampler(makeloglike(loglikearray=logedisplist), 
                                    makeptform(logpriorfunc=logprior), 
                                    ndim=1, 
                                    nlive=nlive, 
                                    sample='rwalk') # Other sampling algorithms tested are 'unif (default) (and rslice but there was not enough dims for it to work)
    
    
    ## Warnings frequently occur due to the use of neffective and 0's within log functions
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.filterwarnings("ignore", category=RuntimeWarning)
    # warnings.filterwarnings("ignore", category=DeprecationWarning)

    
    sampler.run_nested(dlogz=0.05, 
                       print_progress=print_progress, 
                       maxcall=int(5e6))
    
    # Extracting all results from sampler including
    #   - samples used to calculate the evidence values
    #   - the log evidence values
    #   - the weights of the samples used to calculate the evidence values
    #   - the log-likelihood evaluations of the samples used to calculate the evidence values
    #   - and others that can be found in the dynesty documentation.
    
    res = sampler.results
    warnings.filterwarnings("default")
    
    # To get equally weighted samples like MCMC use res.samples_equal()

    return res