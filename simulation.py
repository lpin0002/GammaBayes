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
        
    # try:
    #     lambdaval = float(sys.argv[5])
    # except:
    #     lambdaval = 1.0
    lambdaval = 1.0

    try:
        numcores = int(float(sys.argv[5]))
    except:
        numcores = 10

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

    sigdistsetup = setup_full_fake_signal_dist



    sigdist = sigdistsetup(truelogmass, normeaxis=10**log10eaxis)


    nsig = int(np.round(lambdaval*nevents))
    sigbinnedprior = sigdist(log10emesh, offsetmesh)+logjacob
    flattened_sigbinnedprior = sigbinnedprior.flatten()

    sigresultindices = np.unravel_index(inverse_transform_sampling(flattened_sigbinnedprior, Nsamples=nsig),sigbinnedprior.shape)
    siglogevals = log10eaxis[sigresultindices[1]]
    sigoffsetvals = offsetaxis[sigresultindices[0]]


    signal_log10e_measured = []
    with Pool(numcores) as pool: 
        for result in tqdm(pool.imap(sampleedisp, zip(siglogevals, sigoffsetvals)), 
                            total=len(list(sigoffsetvals)), ncols=100, desc="Creating measured log energy values"):
                signal_log10e_measured.append(log10eaxis[result])

        pool.close() 
    signal_log10e_measured = np.squeeze(np.array(signal_log10e_measured))
    
    # signal_log10e_measured = log10eaxis[np.squeeze([inverse_transform_sampling(edisp(log10eaxis, 
    #                                                                                 logeval, 
    #                                                                                 offsetval)+logjacob, Nsamples=1) for logeval, offsetval in tqdm(zip(siglogevals, sigoffsetvals), total=nsig)])]

    signal_offset_measured = []
    with Pool(numcores) as pool: 
        for result in tqdm(pool.imap(samplepsf, zip(siglogevals, sigoffsetvals)), 
                            total=len(list(sigoffsetvals)), ncols=100, desc="Creating measured offset values"):
                signal_offset_measured.append(offsetaxis[result])

        pool.close() 
    signal_offset_measured = np.squeeze(np.array(signal_offset_measured))
    
    # signal_offset_measured = offsetaxis[np.squeeze([inverse_transform_sampling(psf(offsetaxis, 
    #                                                                             offsetval, 
    #                                                                             logeval), Nsamples=1) for logeval, offsetval in tqdm(zip(siglogevals, sigoffsetvals), total=nsig)])]



    np.save(f"data/{identifier}/{runnum}/truesigsamples.npy", np.array([siglogevals,sigoffsetvals]))
    np.save(f"data/{identifier}/{runnum}/meassigsamples.npy", np.array([signal_log10e_measured,signal_offset_measured]))
    # np.save(f"data/{identifier}/{runnum}/truebkgsamples.npy", bkgsamples)
    # np.save(f"data/{identifier}/{runnum}/measbkgsamples.npy", bkgsamples_measured)
    np.save(f"data/{identifier}/{runnum}/params.npy",         np.array([['lambda', 'Nsamples', 'logmass'],
                                            [lambdaval, nevents, truelogmass]]))

    print("Done simulation.")