
from utils import log10eaxis, makelogjacob, logjacob
import numpy as np
import warnings
import functools
import dynesty.pool as dypool
from scipy import special
import dynesty, warnings


def setup_loglikelihood(cube, edisp_and_jacobian_vals, logsigpriorsetup, logbkgprior, log10eaxis=log10eaxis, logjacob=logjacob):
    log10mass_value = cube[0]
    lambda_value = cube[1]
    
    temp_signal_function = logsigpriorsetup(log10mass_value, normeaxis=10**log10eaxis)
    
    signalnormalisation = special.logsumexp(temp_signal_function(log10eaxis)+logjacob)
    bkgnormalisation = special.logsumexp(logbkgprior(log10eaxis)+logjacob)
    
    bkg_component = np.log(1-lambda_value) + special.logsumexp(logbkgprior(log10eaxis) -bkgnormalisation + edisp_and_jacobian_vals, axis=1)
    sig_component = np.log(lambda_value) + special.logsumexp(temp_signal_function(log10eaxis) - signalnormalisation + edisp_and_jacobian_vals, axis=1)
    

    unsummed_output = np.logaddexp(sig_component, bkg_component)
    return np.sum(unsummed_output)

def ptform(u):
    # For sampler log mass values between 0.1TeV and 100TeV
    u[0] = 3*u[0]-1
    
    # For lambda
    u[1] = u[1]
    
    return u



def brutedynesty(measuredevents, logsigpriorsetup, logbkgprior, logedisp, log10eaxis=log10eaxis, nlive = 500, numcores=10, print_progress=True):
    logjacob = makelogjacob(log10eaxis)
    
    edisp_and_jacobian_vals= logedisp(measuredevents, log10eaxis)+logjacob
    
    norm_vals = np.array([special.logsumexp(logedisp(log10eaxis, log10energyval)+logjacob) for log10energyval in log10eaxis])
    
    
    edisp_and_jacobian_vals = (edisp_and_jacobian_vals - norm_vals)
    

    gaussfull = functools.partial(setup_loglikelihood, edisp_and_jacobian_vals=edisp_and_jacobian_vals,
                                  logsigpriorsetup=logsigpriorsetup, logbkgprior=logbkgprior,log10eaxis=log10eaxis, logjacob=logjacob)

    print(gaussfull((0.0,0.8)))
    
    
    with dypool.Pool(numcores, gaussfull, ptform) as pool:
        # print(dir(pool))
        sampler = dynesty.NestedSampler(
            pool.loglike,
            pool.prior_transform,
            ndim=2, nlive=nlive, bound='multi', # Other bounding methods tested are `balls' and `single'
            pool=pool, queue_size=numcores)

        sampler.run_nested(dlogz=0.01, 
                        print_progress=print_progress, 
                        maxcall=int(1e6))

    
    res = sampler.results
    warnings.filterwarnings("default")
    

    return res

