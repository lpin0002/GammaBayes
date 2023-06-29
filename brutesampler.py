
from utils import log10eaxis, makelogjacob, logjacob, offsetaxis, setup_full_fake_signal_dist, log10eaxistrue, offsetaxistrue, logjacobtrue
import numpy as np
import warnings
import functools
import dynesty.pool as dypool
from scipy import special
import dynesty, warnings


log10emeshtrue, offsetmeshtrue = np.meshgrid(log10eaxistrue, offsetaxistrue)


def setup_loglikelihood(cube, logirfvals, specfunc, bkgmargvals, log10emeshtrue=log10emeshtrue, offsetmeshtrue=offsetmeshtrue, logjacobtrue=logjacobtrue):
    log10mass_value = cube[0]
    lambda_value = cube[1]
    
    
    signalpriorfunc = setup_full_fake_signal_dist(log10mass_value, specfunc=specfunc)
    
    sigpriorvalues = np.squeeze(signalpriorfunc(log10emeshtrue, offsetmeshtrue))
    
    signalnormalisation = special.logsumexp(sigpriorvalues+logjacobtrue)

    # Integration/nuisance parameter marginalisation step
    bkg_component = np.log(1-lambda_value) + bkgmargvals
    sig_component = np.log(lambda_value) + special.logsumexp(sigpriorvalues - signalnormalisation + logirfvals + logjacobtrue, axis=(1,2))
    

    summed_output = np.nansum(np.logaddexp(sig_component, bkg_component))
    
    del bkg_component
    del sig_component
    del sigpriorvalues
    
    return summed_output

def ptform(u):
    # For sampler log mass values between 100 TeV and what ever the lowest energy considered is
        # You would not be able to have a dark matter event if the mass is below the energy range
    u[0] = (2-log10eaxis[0])*u[0]+log10eaxis[0]
    
    # For lambda
    u[1] = u[1]
    
    return u



def brutedynesty(specfunc, logbkgprior, logirfvals, log10eaxistrue=log10eaxistrue, offsetaxistrue=offsetaxistrue, nlive = 300, numcores=10, print_progress=True):
    
    log10emeshtrue, offsetmeshtrue = np.meshgrid(log10eaxistrue, offsetaxistrue)

    logjacobtrue = makelogjacob(log10eaxistrue)    
    
    logbkgpriorvalues = np.squeeze(logbkgprior(log10emeshtrue, offsetmeshtrue))
    
    
    bkgnormalisation = special.logsumexp(logbkgpriorvalues+logjacobtrue)
    
    logbkgpriorvalues = logbkgpriorvalues - bkgnormalisation
    
    bkgmargvals = special.logsumexp(logbkgpriorvalues + logirfvals + logjacobtrue, axis=(1,2))
    
    

    gaussfull = functools.partial(setup_loglikelihood, logirfvals=logirfvals,
                                  specfunc=specfunc, bkgmargvals=bkgmargvals,log10emeshtrue=log10emeshtrue, offsetmeshtrue=offsetmeshtrue, logjacobtrue=logjacobtrue)

    
    
    with dypool.Pool(numcores, gaussfull, ptform) as pool:
        # print(dir(pool))
        sampler = dynesty.NestedSampler(
            pool.loglike,
            pool.prior_transform,
            ndim=2, nlive=nlive, bound='single', # Other bounding methods tested are `balls' and `single'
            pool=pool, queue_size=numcores)

        sampler.run_nested(dlogz=0.3, 
                        print_progress=print_progress, 
                        maxcall=int(1e6))

    
    res = sampler.results
    warnings.filterwarnings("default")
    

    return res

