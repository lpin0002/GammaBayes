import numpy as np
from scipy import integrate, interpolate, special, stats
from multiprocessing import Pool
import dynesty
import dynesty.pool as dypool
import functools


def log_pt_recycling(lambdaval, proposalresults, bkgresults, logproposalprior, logtargetprior):
    prod = 0

    for proposalres, bkgres in zip(proposalresults, bkgresults):
        # The background contributino to the hyperparameter likelihood
        logbkgcomp                 = np.log(1 - lambdaval) + bkgres.logz[-1]

        # Accessing the __equally weighted__ samples from the marginalisation with the proposal prior
        propposal_posteriorsamples    = proposalres.samples_equal()

        # Probability values for log_10(E) samples
        logtargetprior_comp     = logtargetprior(propposal_posteriorsamples)
        logproposalprior_comp   = logproposalprior(propposal_posteriorsamples)

        # Doing the division
        logfrac                 = logtargetprior_comp - logproposalprior_comp

        # The signal contribution to the hyperparameter likelihood
        logsigcomp                 = np.log(lambdaval) + proposalres.logz[-1] + special.logsumexp(logfrac) -np.log(propposal_posteriorsamples.shape[0])


        prod += np.logaddexp(logbkgcomp, logsigcomp)

    return prod



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
    return u



def runrecycle(propres, bkgres, logpropprior, logtargetpriorsetup, recyclingcores = 10, nlive = 2000, print_progress=False):

    # Setting up the likelihood. Which in our case is a product of the point spread function and energy dispersion for the CTA

    inputloglikefunc = functools.partial(inputloglike, propresults=propres, bkgresults=bkgres, logproposalprior=logpropprior, logtargetpriorsetup=logtargetpriorsetup)
        
    ptprior = ptform

    with dypool.Pool(recyclingcores, inputloglikefunc, ptprior) as pool:
        # print(dir(pool))
        sampler = dynesty.NestedSampler(
            pool.loglike,
            pool.prior_transform,
            ndim=2, nlive=nlive, bound='multi', pool=pool, queue_size=recyclingcores, update_interval=1.0)

        sampler.run_nested(dlogz=0.1, print_progress=print_progress)

    # Extracting the results from the sampler
    results = sampler.results

    # To get equally weighted posterior samples like MCMC use res.samples_equal()

    return results