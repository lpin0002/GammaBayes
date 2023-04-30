import numpy as np
from scipy import integrate, interpolate, special, stats
from multiprocessing import Pool
import dynesty
import dynesty.pool as dypool
import functools


def singlesamplemixture(proposalresult, bkgresult, lambdaval, logproposalprior, logtargetprior):
    bkgcomp = np.log(1 - lambdaval) + bkgresult.logz[-1]
    logfrac = special.logsumexp(logtargetprior(proposalresult.samples_equal()) - logproposalprior(proposalresult.samples_equal()))
    sigcomp = np.log(lambdaval) + proposalresult.logz[-1] + logfrac-np.log(proposalresult.samples_equal().shape[0])
    value = np.logaddexp(bkgcomp, sigcomp)
    
    return value

def log_pt_recycling(lambdaval, proposalresults, bkgresults, logproposalprior, logtargetprior):
    partialmixturefunc = functools.partial(singlesamplemixture, lambdaval=lambdaval, logproposalprior=logproposalprior, logtargetprior=logtargetprior)
    prodlist = [partialmixturefunc(proposalresult, bkgresult) for proposalresult, bkgresult in zip(proposalresults, bkgresults)]

    return np.sum(prodlist)



def inputloglike(cube, propresults, bkgresults, logproposalprior, logtargetpriorsetup):
    logmassval = cube[0]
    lambdaval = cube[1]

    logtargetprior = logtargetpriorsetup(logmassval)

    output = log_pt_recycling(lambdaval, propresults, bkgresults, logproposalprior, logtargetprior)
    return output



def ptform(u):
    # log mass vals to go from 10 GeV to 1 PeV
    u[0] = 5*u[0]-2

    # lambdavals already should be uniform between 0 and 1
    u[1] = u[1]
    return u[0], u[1]



def runrecycle(propres, bkgres, logpropprior, logtargetpriorsetup, recyclingcores = 10, nlive = 1000, print_progress=False):

    # Setting up the likelihood. Which in our case is a product of the point spread function and energy dispersion for the CTA

    inputloglikefunc = functools.partial(inputloglike, propresults=propres, bkgresults=bkgres, logproposalprior=logpropprior, logtargetpriorsetup=logtargetpriorsetup)
        
    ptprior = ptform

    with dypool.Pool(recyclingcores, inputloglikefunc, ptprior) as pool:
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