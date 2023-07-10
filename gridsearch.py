from BFCalc.BFInterp import DM_spectrum_setup
from BFCalc.createspectragrids import singlechannel_diffflux, getspectrafunc, darkmatterdoubleinput
from utils3d import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy import units as u
from scipy import special,stats
from matplotlib import cm
from tqdm.autonotebook import tqdm as notebook_tqdm
import os, sys, time, functools
from multiprocessing import Pool, freeze_support
import multiprocessing
sys.path.append("BFCalc")


if __name__=="__main__":
    

    try:
        identifier = sys.argv[1]
    except:
        raise Exception("You have not input a valid identifier. Please review the name of the stemfolder for your runs.")
    
    try:
        nbinslogmass = int(sys.argv[2])
    except:
        nbinslogmass = 11

    try:
        nbinslambda = int(sys.argv[3])
    except:
        nbinslambda = 11
    
    try:
        numcores = int(sys.argv[4])
    except:
        numcores = 10
        
    signalspecfunc = darkmatterdoubleinput
    lontrue_mesh_nuisance, logetrue_mesh_nuisance, lattrue_mesh_nuisance = np.meshgrid(spatialaxistrue, log10eaxistrue, spatialaxistrue)


    
    
    
    #Generating various paths to files within stem directory
    currentdirecyory = os.getcwd()
    stemdirectory = currentdirecyory+f'/data/{identifier}'
    print("\nstem directory: ", stemdirectory, '\n')
    
    rundirs = [x[0] for x in os.walk(stemdirectory)][1:]
    print("number of run directories: ", len(rundirs), '\n')
    
    firstrundirectory = rundirs[0]
    
    
    
    # Loading in values
    irfproblist = np.load(f"{stemdirectory}/irfproblist.npy")
    # logbackgroundprior = np.load(f"{firstrundirectory}/logbackgroundprior.npy")
    truelambda, Nsamples, truelogmassval = np.load(f"{firstrundirectory}/params.npy")[1,:]
    Nsamples = int(Nsamples)
    truelogmassval = float(truelogmassval)
    truelambda = float(truelambda)


    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # Signal Nuisance Parameter Marginalisation


    sigdistsetup = setup_full_fake_signal_dist
    # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
    np.seterr(divide='ignore', invalid='ignore')




    nsig = int(round(truelambda*Nsamples))


    # Generating the range of log mass values to be tested
    logmasswindowwidth      = 5/np.sqrt(nsig)

    logmasslowerbound       = truelogmassval-logmasswindowwidth
    logmassupperbound       = truelogmassval+logmasswindowwidth

    if logmasslowerbound<log10eaxis[0]:
        logmasslowerbound = log10eaxis[0]
    if logmassupperbound>2:
        logmassupperbound = 2
        
    logmassrange            = np.linspace(logmasslowerbound, logmassupperbound, nbinslogmass) 


    # Generating the range of lambda values to be tested
    lambdawindowwidth      = 5/np.sqrt(Nsamples)
    
    lambdalowerbound       = truelambda-lambdawindowwidth
    lambdaupperbound       = truelambda+lambdawindowwidth
    

    if lambdalowerbound<0:
        lambdalowerbound = 0
    if lambdaupperbound>1:
        lambdaupperbound = 1

    lambdarange            = np.linspace(lambdalowerbound, lambdaupperbound, nbinslambda) 
    
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # Nuisance Parameter Marginalisation with Signal Prior

    tempsigmargfunction = functools.partial(evaluateformass, irfvals=irfproblist, specfunc=signalspecfunc, lontrue_mesh_nuisance=lontrue_mesh_nuisance, logetrue_mesh_nuisance=logetrue_mesh_nuisance, lattrue_mesh_nuisance=lattrue_mesh_nuisance )

    signal_log_marginalisationvalues = []

    with Pool(numcores) as pool: 
            
        for result in notebook_tqdm(pool.imap(tempsigmargfunction, logmassrange), total=int(len(list(logmassrange))), ncols=100, desc="Calculating signal marginalisations..."):
                signal_log_marginalisationvalues.append(result)

        pool.close() 
        
        
        
    signal_log_marginalisationvalues = np.array(signal_log_marginalisationvalues)
    print(f"Shape of array containing the results of marginalising the nuisance parameters with the signal prior: {signal_log_marginalisationvalues.shape}")
    
    
    
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # Nuisance Parameter Marginalisation with Background Prior
    
    bkgpriorarray  = bkgdist(logetrue_mesh_nuisance, lontrue_mesh_nuisance, lattrue_mesh_nuisance)
    bkgpriorarray = bkgpriorarray - special.logsumexp(bkgpriorarray.T+logjacobtrue)

    bkgmargvals = [special.logsumexp(bkgpriorarray.T+irfarray.T+logjacobtrue) for irfarray in irfproblist]
    bkgmargvals = np.array(bkgmargvals)
    
    print(f"Shape of array containing the results of marginalising the nuisance parameters with the background prior: {bkgmargvals.shape}")

    
    
    ##########################################################################################################################################################################
    ##########################################################################################################################################################################
    # Combine nuisance parameter marginalisation results to calculate (log) posterior
    
    

    log_posterior = []

    for lambdaval in notebook_tqdm(lambdarange, total=lambdarange.shape[0]):
        log_posterior.append([np.sum(np.logaddexp(np.log(lambdaval)+signal_log_marginalisationvalues[logmassindex,:], np.log(1-lambdaval)+bkgmargvals)) for logmassindex in range(len(list(logmassrange)))])


    unnormalised_log_posterior = np.array(log_posterior)


    print(f"Shape of the array containing the log of the unnormalised posterior: {unnormalised_log_posterior.shape}")

    
    
    
    np.save(f'{stemdirectory}/lambdarange_direct.npy',lambdarange)
    np.save(f'{stemdirectory}/logmassrange_direct.npy',logmassrange)
    np.save(f'{stemdirectory}/unnormalised_logposterior_direct.npy', unnormalised_log_posterior)
    
    
    
    
    
    
    # If this is not done, some of the below nice functionality fails (e.g. mean and sigma calculations)
    log_posterior = log_posterior-special.logsumexp(log_posterior)
    
    
    
    
    
    from utils3d import confidence_ellipse
    from scipy.stats import norm

    import time

    colormap = cm.get_cmap('Blues_r', 4)

    fig, ax = plt.subplots(2,2, dpi=100, figsize=(10,8))
    plt.suptitle(f"Nevents= {Nsamples}", size=24)

    # Upper left plot
    logmass_logposterior = special.logsumexp(log_posterior, axis=0)

    normalisedlogmassposterior = np.exp(logmass_logposterior-special.logsumexp(logmass_logposterior))

    cdflogmassposterior = np.cumsum(normalisedlogmassposterior)
    mean = logmassrange[np.abs(norm.cdf(0)-cdflogmassposterior).argmin()]
    zscores = [-3, -2,-1,1,2, 3]
    logmasspercentiles = []
    for zscore in zscores:
        logmasspercentiles.append(logmassrange[np.abs(norm.cdf(zscore)-cdflogmassposterior).argmin()])


    ax[0,0].plot(logmassrange,normalisedlogmassposterior, c='tab:green')

    ax[0,0].axvline(mean, c='tab:green', ls=':')


    for o, percentile in enumerate(logmasspercentiles):
                color = colormap(np.abs(zscores[o])/4-0.01)

                ax[0,0].axvline(percentile, c=color, ls=':')
    ax[0,0].axvline(truelogmassval, ls='--', color="tab:orange")


    if min(mean - logmasspercentiles)>log10eaxistrue[1]-log10eaxistrue[0]:
        for logetrueval in log10eaxistrue:
            ax[0,0].axvline(logetrueval, c='forestgreen', alpha=0.3)
    ax[0,0].set_ylim([0, None])
    ax[0,0].set_xlim([logmassrange[0], logmassrange[-1]])

    # Upper right plot
    ax[0,1].axis('off')


    # Lower left plot
    # ax[1,0].pcolormesh(logmassrange, lambdarange, np.exp(normalisedlogposterior).T, cmap='Blues')
    ax[1,0].pcolormesh(logmassrange, lambdarange, np.exp(log_posterior))
    ax[1,0].axvline(truelogmassval, c='tab:orange')
    ax[1,0].axhline(truelambda, c='tab:orange')
    ax[1,0].set_xlabel(r'$log_{10}$ mass [TeV]')
    ax[1,0].set_ylabel(r'$\lambda$')

    ax[1,0].set_ylim([lambdarange[0], lambdarange[-1]])
    ax[1,0].set_xlim([logmassrange[0], logmassrange[-1]])

    confidence_ellipse(logmassrange, lambdarange, np.exp(log_posterior), ax[1,0], n_std=3.0, linewidth=1.5)
    confidence_ellipse(logmassrange, lambdarange, np.exp(log_posterior), ax[1,0], n_std=2.0, linewidth=1.5)
    confidence_ellipse(logmassrange, lambdarange, np.exp(log_posterior), ax[1,0], n_std=1.0, linewidth=1.5)


    lambda_logposterior = special.logsumexp(log_posterior, axis=1)

    normalisedlambdaposterior = np.exp(lambda_logposterior-special.logsumexp(lambda_logposterior))

    cdflambdaposterior = np.cumsum(normalisedlambdaposterior)
    meanlambda = lambdarange[np.abs(norm.cdf(0)-cdflambdaposterior).argmin()]
    lambdapercentiles = []
    for zscore in zscores:
        lambdapercentiles.append(lambdarange[np.abs(norm.cdf(zscore)-cdflambdaposterior).argmin()])


    ax[1,1].plot(lambdarange,normalisedlambdaposterior, c='tab:green')

    ax[1,1].axvline(meanlambda, c='tab:green', ls=':')


    for o, percentile in enumerate(lambdapercentiles):
                color = colormap(np.abs(zscores[o])/4-0.01)

                ax[1,1].axvline(percentile, c=color, ls=':')
    ax[1,1].axvline(truelambda, ls='--', color="tab:orange")
    ax[1,1].set_xlabel(r'$\lambda$')
    ax[1,1].set_ylim([0, None])


    plt.savefig(time.strftime(f"Figures/TestFigures/{Nsamples}events_lm{truelogmassval}_l{truelambda}_%m%d_%H%M.pdf"))
    plt.show()





