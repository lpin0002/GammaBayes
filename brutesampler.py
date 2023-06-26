
from utils import log10eaxis, makelogjacob, logjacob, offsetaxis, setup_full_fake_signal_dist
import numpy as np
import warnings
import functools
import dynesty.pool as dypool
from scipy import special
import dynesty, warnings


def setup_loglikelihood(cube, logirfvals, specsetup, logbkgpriorarray, log10eaxis=log10eaxis, offsetaxis=offsetaxis, logjacob=logjacob):
    log10mass_value = cube[0]
    lambda_value = cube[1]
    
    
    signalpriorfunc = setup_full_fake_signal_dist(log10mass_value, specsetup=specsetup, normeaxis=10**log10eaxis)
    
    signalpriorvals = []
    for offsetval in offsetaxis:
        signalpriorvals.append(signalpriorfunc(log10eaxis, offsetval))
    signalpriorvals = np.squeeze(np.array(signalpriorvals))
    
    
    signalnormalisation = special.logsumexp(signalpriorvals+logjacob)

    # Integration/nuisance parameter marginalisation step
    bkg_component = np.log(1-lambda_value) + special.logsumexp(logbkgpriorarray[np.newaxis,:] + logirfvals, axis=(1,2))
    sig_component = np.log(lambda_value) + special.logsumexp(signalpriorvals[np.newaxis,:] - signalnormalisation + logirfvals, axis=(1,2))
    

    unsummed_output = np.logaddexp(sig_component, bkg_component)
    summedoutput=np.nansum(unsummed_output)
    
    
    return summedoutput

def ptform(u):
    # For sampler log mass values between 100 TeV and what ever the lowest energy considered is
        # You would not be able to have a dark matter event if the mass is below the energy range
    u[0] = (2-log10eaxis[0])*u[0]+log10eaxis[0]
    
    # For lambda
    u[1] = u[1]
    
    return u



def brutedynesty(specsetup, logbkgprior, logirfvals, log10eaxis=log10eaxis, offsetaxis=offsetaxis, nlive = 500, numcores=10, print_progress=True):
    logjacob = makelogjacob(log10eaxis)    
    
    logbkgpriorarray = []
    for ii, logeval in enumerate(log10eaxis):
        singlerow = []
        # for ii, offsetval in enumerate(offsetaxis):
        singlerow = logbkgprior(logeval, offsetaxis)
        logbkgpriorarray.append(singlerow)
    logbkgpriorarray = np.squeeze(np.array(logbkgpriorarray))
    
    
    bkgnormalisation = special.logsumexp(logbkgpriorarray.T+logjacob)
    
    logbkgpriorarray = logbkgpriorarray.T - bkgnormalisation
    
    

    gaussfull = functools.partial(setup_loglikelihood, logirfvals=logirfvals,
                                  specsetup=specsetup, logbkgpriorarray=logbkgpriorarray,log10eaxis=log10eaxis, offsetaxis=offsetaxis, logjacob=logjacob)

    
    
    with dypool.Pool(numcores, gaussfull, ptform) as pool:
        # print(dir(pool))
        sampler = dynesty.NestedSampler(
            pool.loglike,
            pool.prior_transform,
            ndim=2, nlive=nlive, bound='multi', # Other bounding methods tested are `balls' and `single'
            pool=pool, queue_size=numcores)

        sampler.run_nested(dlogz=0.1, 
                        print_progress=print_progress, 
                        maxcall=int(1e6))

    
    res = sampler.results
    warnings.filterwarnings("default")
    

    return res

