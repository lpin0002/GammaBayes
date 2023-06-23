from scipy import integrate, special, interpolate, stats
import os, sys, time, random, numpy as np, matplotlib.pyplot as plt, warnings
from tqdm import tqdm
from utils import inverse_transform_sampling, log10eaxis, makedist, edisp, psf, bkgdist, eaxis_mod, eaxis, logjacob, setup_full_fake_signal_dist, offsetaxis, log10emesh, offsetmesh
from BFCalc.BFInterp import DM_spectrum_setup
# Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
np.seterr(divide='ignore', invalid='ignore')
# Check that jacobian for this script is correct
import functools
from multiprocessing import Pool, freeze_support
import multiprocessing


def sampleedisp(measuredvals):
    logeval, offsetval = measuredvals
    return inverse_transform_sampling(edisp(log10eaxis, logeval, offsetval)+logjacob, Nsamples=1)


def samplepsf(measuredvals):
    logeval, offsetval = measuredvals
    return inverse_transform_sampling(psf(offsetaxis, offsetval, logeval), Nsamples=1)


if __name__=="__main__":

    try:
        identifier = sys.argv[1]
    except:
        identifier = time.strftime("%d%m%H")

    try:
        runnum = int(sys.argv[2])
    except:
        runnum = 1

    try:
        nevents = int(sys.argv[3])
    except:
        nevents = 10

    try:
        truelogmass = float(sys.argv[4]) 
    except:
        truelogmass = 0
        
    try:
        lambdaval = float(sys.argv[5])
    except:
        lambdaval = 0.5

    # try:
    #     numcores = int(float(sys.argv[6]))
    # except:
    #     numcores = 10

    try:
        os.mkdir('data')
    except:
        print("data folder already exists (good)")
    try:
        os.mkdir(f'data/{identifier}')
    except:
        print("Stem Folder Already Exists (good)")
    try:
        os.mkdir(f'data/{identifier}/{runnum}')
    except:
        raise Exception(f"The folder data/{identifier}/{runnum} already exists, stopping computation so files are not accidentally overwritten.")

    log10emesh, offsetmesh = np.meshgrid(log10eaxis, offsetaxis)
    Nsamples=nevents
    truelambda = lambdaval
    truelogmassval = truelogmass
    nsig = int(round(truelambda*Nsamples))
    nbkg = int(round((1-truelambda)*Nsamples))
    
    
    sigpriorvalues = []

    for ii, logeval in enumerate(log10eaxis):
        singlerow = setup_full_fake_signal_dist(truelogmassval, specsetup=DM_spectrum_setup, normeaxis=10**log10eaxis)(logeval, offsetaxis)
        sigpriorvalues.append(singlerow)
    sigpriorvalues = np.array(sigpriorvalues)
    sigpriorvalues.shape
    
    
    bkgpriorvalues = []

    for ii, logeval in enumerate(log10eaxis):
        singlerow = []
        # for ii, offsetval in enumerate(offsetaxis):
        singlerow = bkgdist(logeval, offsetaxis)
        bkgpriorvalues.append(singlerow)
    bkgpriorvalues = np.squeeze(np.array(bkgpriorvalues))
    
    
    sigbinnedprior = sigpriorvalues.T+logjacob
    flattened_sigbinnedprior = sigbinnedprior.flatten()


    bkgbinnedprior = bkgpriorvalues.T+logjacob
    
    bkgnormalisation = special.logsumexp(bkgbinnedprior)
    bkgbinnedprior = bkgbinnedprior - bkgnormalisation
    
    flattened_bkgbinnedprior = bkgbinnedprior.flatten()
    



    sigresultindices = np.unravel_index(inverse_transform_sampling(flattened_sigbinnedprior, Nsamples=nsig),sigbinnedprior.shape)
    siglogevals = log10eaxis[sigresultindices[1]]
    sigoffsetvals = offsetaxis[sigresultindices[0]]
    
    
    signal_log10e_measured = log10eaxis[np.squeeze([inverse_transform_sampling(edisp(log10eaxis, logeval, offsetval)+logjacob, Nsamples=1) for logeval, offsetval in tqdm(zip(siglogevals, sigoffsetvals), total=nsig)])]

    signal_offset_measured = offsetaxis[np.squeeze([inverse_transform_sampling(psf(offsetaxis, offsetval, logeval), Nsamples=1) for logeval, offsetval in tqdm(zip(siglogevals, sigoffsetvals), total=nsig)])]

    bkgresultindices = np.unravel_index(inverse_transform_sampling(flattened_bkgbinnedprior, Nsamples=nbkg),bkgbinnedprior.shape)
    bkglogevals = log10eaxis[bkgresultindices[1]]
    bkgoffsetvals = offsetaxis[bkgresultindices[0]]
    
    bkg_log10e_measured = log10eaxis[np.squeeze([inverse_transform_sampling(edisp(log10eaxis, logeval, offsetval)+logjacob, Nsamples=1) for logeval, offsetval in tqdm(zip(bkglogevals, bkgoffsetvals), total=nbkg)])]
    bkg_offset_measured = offsetaxis[np.squeeze([inverse_transform_sampling(psf(offsetaxis, offsetval, logeval), Nsamples=1) for logeval, offsetval in tqdm(zip(bkglogevals, bkgoffsetvals), total=nbkg)])]


    np.save(f"data/{identifier}/{runnum}/truesigsamples.npy", np.array([siglogevals,sigoffsetvals]))
    np.save(f"data/{identifier}/{runnum}/meassigsamples.npy", np.array([signal_log10e_measured,signal_offset_measured]))
    np.save(f"data/{identifier}/{runnum}/truebkgsamples.npy", np.array([bkglogevals,bkgoffsetvals]))
    np.save(f"data/{identifier}/{runnum}/measbkgsamples.npy", np.array([bkg_log10e_measured,bkg_offset_measured]))
    np.save(f"data/{identifier}/{runnum}/params.npy",         np.array([['lambda', 'Nsamples', 'logmass'],
                                            [lambdaval, nevents, truelogmass]]))
    np.save(f"data/{identifier}/{runnum}/backgroundprior.npy",bkgbinnedprior)

    print("Done simulation.")