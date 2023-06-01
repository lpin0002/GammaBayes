import numpy as np
from scipy import integrate, interpolate, special, stats
from multiprocessing import Pool
import dynesty
import dynesty.pool as dypool
import functools
from utils import log10eaxis, logjacob, makelogjacob

def singlesamplemixture(proposallogzresult, proposalmargsamplelist, bkglogzresult, lambdaval, logproposalprior, logtargetprior, log10eaxis=log10eaxis):
    
    # Log of the background component of the mixture
    logbkgcomponent = np.log(1 - lambdaval) + bkglogzresult
    
    
    logsigvals = np.squeeze(logtargetprior(proposalmargsamplelist))
    
    # Triple checking that the target prior is normalised
    logsignorm = special.logsumexp(logtargetprior(log10eaxis)+makelogjacob(log10eaxis))
    
    # Calculating the sum of the fraction of the value of the target prior of the proposal prior for an event
    logfrac = special.logsumexp(logsigvals - logsignorm - np.squeeze(logproposalprior(proposalmargsamplelist)))
    
    
    
    # Calculating the rest of the signal component of the mixture model
    logsignalcomponent = np.log(lambdaval) + proposallogzresult + logfrac-np.log(proposalmargsamplelist.shape[0])
    
    
    # Adding the two components
    value = np.logaddexp(logbkgcomponent, logsignalcomponent)
    
    
    return value

def log_pt_recycling(lambdaval, proposallogzresults, proposalmargsamples, bkglogevidencevalues, logproposalprior, logtargetprior, log10eaxis=log10eaxis):
    
    # Input the common keyword arguments for the samples from marginalising over the nuisance parameters with the proposal prior to reweight with the target prior
    mixture_singlesampleinput = functools.partial(singlesamplemixture, lambdaval=lambdaval, logproposalprior=logproposalprior, logtargetprior=logtargetprior, log10eaxis=log10eaxis)
    
    # Need to vectorise
    listof_logprobabilityvalues = [mixture_singlesampleinput(proposallogzresult, proposalmargsamplelist, bkglogzresult) for proposallogzresult, proposalmargsamplelist, bkglogzresult in zip(proposallogzresults, proposalmargsamples, bkglogevidencevalues)]
    summedloglike = np.nansum(listof_logprobabilityvalues)
    
    # print("summedloglike: ", summedloglike)
    return summedloglike



def inputloglike(cube, log10eaxis, proposallogzresults, proposalmargsamples, bkglogevidencevalues, logproposalprior, logtargetpriorsetup):
    
    # Extracting the log mass and value of lambda from the prior transform
    logmassval = cube[0]
    lambdaval = cube[1]
    
    # Setting up the target prior for a specific mass and energy axis for normalisation
    logtargetprior = logtargetpriorsetup(logmassval, normeaxis=10**log10eaxis)
    # Generating the log-likelihood
    loglikelihoodoutput = log_pt_recycling(lambdaval, proposallogzresults, proposalmargsamples, bkglogevidencevalues, logproposalprior, logtargetprior, log10eaxis=log10eaxis)
    
    
    
    return np.squeeze(loglikelihoodoutput)



def ptform(u):
    # log mass [TeV] (currently from 0.1 TeV to 10 TeV)
    logmassval = 3.0*u[0]-1.0

    # Fraction of signal to total events
    lambdavals = u[1]
    return logmassval, lambdavals




def runrecycle(propresults, bkgmargresults, logpropprior, logtargetpriorsetup, log10eaxis, recyclingcores = 10, nlive = 300, print_progress=False):
    
    # Extracting the key aspects needed from the nested sampler
    bkglogevidencevalues = [bkgmargresult.logz[-1] for bkgmargresult in bkgmargresults]  # The log evidence values from marginalising with the background
    proposallogzresults = [propresult.logz[-1] for propresult in propresults] # The log evidence values from marginalisaing with the proposal
    
    proposalmargsamples = [np.squeeze(propresult.samples_equal()) for propresult in propresults] # Extracting the samples representing of the nuisance parameter posterior with the proposal prior
    
    
    
    
    
    # Setting up the likelihood. Which in our case is the energy dispersion for the CTA
    inputloglikefunc = functools.partial(inputloglike, proposallogzresults=proposallogzresults, proposalmargsamples=proposalmargsamples, bkglogevidencevalues=bkglogevidencevalues, 
                                         logproposalprior=logpropprior, logtargetpriorsetup=logtargetpriorsetup,
                                         log10eaxis=log10eaxis)
    
    
    with dypool.Pool(recyclingcores, inputloglikefunc, ptform) as pool:
        # print(dir(pool))
        sampler = dynesty.NestedSampler(
            pool.loglike,
            pool.prior_transform,
            ndim=2, nlive=nlive, bound='multi', # Other bounding methods tested are `balls' and `single'
            pool=pool, queue_size=recyclingcores)

        sampler.run_nested(dlogz=0.05, print_progress=print_progress)

    # Extracting the results from the sampler
    results = sampler.results

    # To get equally weighted posterior samples like MCMC use res.samples_equal()

    return results