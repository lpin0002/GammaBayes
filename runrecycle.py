import numpy as np
from scipy import integrate, interpolate, special, stats
from multiprocessing import Pool
import dynesty
import dynesty.pool as dypool
import functools


def singlesamplemixture(proposalresult, bkgresult, lambdaval, logproposalprior, logtargetprior):
    logbkgcomponent = np.log(1 - lambdaval) + bkgresult.logz[-1]
    logfrac = special.logsumexp(logtargetprior(proposalresult.samples_equal()) - logproposalprior(proposalresult.samples_equal()))
    logsignalcomponent = np.log(lambdaval) + proposalresult.logz[-1] + logfrac-np.log(proposalresult.samples_equal().shape[0])
    value = np.logaddexp(logbkgcomponent, logsignalcomponent)
    
    return value

def log_pt_recycling(lambdaval, proposalresults, bkgresults, logproposalprior, logtargetprior):
    
    mixture_onlyresultsinput = functools.partial(singlesamplemixture, lambdaval=lambdaval, logproposalprior=logproposalprior, logtargetprior=logtargetprior)
    listof_logprobabilityvalues = [mixture_onlyresultsinput(proposalresult, bkgresult) for proposalresult, bkgresult in zip(proposalresults, bkgresults)]

    return np.sum(listof_logprobabilityvalues)



def inputloglike(cube, propresults, bkgresults, logproposalprior, logtargetpriorsetup):
    logmassval = cube[0]
    lambdaval = cube[1]

    logtargetprior = logtargetpriorsetup(logmassval)

    output = log_pt_recycling(lambdaval, propresults, bkgresults, logproposalprior, logtargetprior)
    return output



def ptform(u):
    # log mass [TeV]
    logmassval = 5*u[0]-2

    lambdavals = u[1]
    return logmassval, lambdavals



def runrecycle(propres, bkgres, logpropprior, logtargetpriorsetup, recyclingcores = 10, nlive = 2000, print_progress=False):

    # Setting up the likelihood. Which in our case is a product of the point spread function and energy dispersion for the CTA
    inputloglikefunc = functools.partial(inputloglike, propresults=propres, bkgresults=bkgres, logproposalprior=logpropprior, logtargetpriorsetup=logtargetpriorsetup)
    
    with dypool.Pool(recyclingcores, inputloglikefunc, ptform) as pool:
        # print(dir(pool))
        sampler = dynesty.NestedSampler(
            pool.loglike,
            pool.prior_transform,
            ndim=2, nlive=nlive, bound='single', pool=pool, queue_size=recyclingcores)

        sampler.run_nested(dlogz=0.1, print_progress=print_progress)

    # Extracting the results from the sampler
    results = sampler.results

    # To get equally weighted posterior samples like MCMC use res.samples_equal()

    return results