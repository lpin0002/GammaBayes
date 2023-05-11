import numpy as np
from scipy import integrate, interpolate, special, stats
from multiprocessing import Pool
import dynesty
import dynesty.pool as dypool
import functools

def singlesamplemixture(proposallogzresult, proposalmargsamplelist, bkglogzresult, lambdaval, logproposalprior, logtargetprior):
    logbkgcomponent = np.log(1 - lambdaval) + bkglogzresult
    logfrac = special.logsumexp(logtargetprior(proposalmargsamplelist) - logproposalprior(proposalmargsamplelist))
    logsignalcomponent = np.log(lambdaval) + proposallogzresult + logfrac-np.log(proposalmargsamplelist.shape[0])
    
    value = np.logaddexp(logbkgcomponent, logsignalcomponent)
    return value

def log_pt_recycling(lambdaval, proposallogzresults, proposalmargsamples, bkglogevidencevalues, logproposalprior, logtargetprior):
    
    mixture_singlesampleinput = functools.partial(singlesamplemixture, lambdaval=lambdaval, logproposalprior=logproposalprior, logtargetprior=logtargetprior)
    listof_logprobabilityvalues = [mixture_singlesampleinput(proposallogzresult, proposalmargsamplelist, bkglogzresult) for proposallogzresult, proposalmargsamplelist, bkglogzresult in zip(proposallogzresults, proposalmargsamples, bkglogevidencevalues)]

    return np.sum(listof_logprobabilityvalues)



def inputloglike(cube, log10eaxis, proposallogzresults, proposalmargsamples, bkglogevidencevalues, logproposalprior, logtargetpriorsetup):
    logmassval = cube[0]
    lambdaval = cube[1]

    logtargetprior = logtargetpriorsetup(logmassval, normeaxis=10**log10eaxis)

    output = log_pt_recycling(lambdaval, proposallogzresults, proposalmargsamples, bkglogevidencevalues, logproposalprior, logtargetprior)
    
    return output



def ptform(u):
    # log mass [TeV]
    logmassval = 3.*u[0]-1.

    lambdavals = u[1]
    return logmassval, lambdavals




def runrecycle(propresults, bkgmargresults, logpropprior, logtargetpriorsetup, log10eaxis, recyclingcores = 10, nlive = 200, print_progress=False):
    
    bkglogevidencevalues = [bkgmargresult.logz[-1] for bkgmargresult in bkgmargresults]
    proposallogzresults = [propresult.logz[-1] for propresult in propresults]
    proposalmargsamples = [propresult.samples_equal() for propresult in propresults]
    # Setting up the likelihood. Which in our case is a product of the point spread function and energy dispersion for the CTA
    inputloglikefunc = functools.partial(inputloglike, proposallogzresults=proposallogzresults, proposalmargsamples=proposalmargsamples, bkglogevidencevalues=bkglogevidencevalues, 
                                         logproposalprior=logpropprior, logtargetpriorsetup=logtargetpriorsetup,
                                         log10eaxis=log10eaxis)
    
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