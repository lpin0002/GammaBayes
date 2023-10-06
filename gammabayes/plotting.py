import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import time
from matplotlib import cm
import sys, os
from scipy.special import logsumexp
from .utils.utils import read_config_file

import os






def plot_posterior(log_posterior, xi_range, logmassrange, truevals=None, identifier=None, Nevents=None, 
                   saveplot=True, save_directory=None, credible_levels=None):
    if save_directory is None:
        if identifier is None:
            save_directory = time.strftime(f"posteriorplot_%m%d_%H.pdf")
        else:
            save_directory = time.strftime(f"data/{identifier}/posteriorplot_%m%d_%H.pdf")
    
    colormap = cm.get_cmap('Blues_r', 4)

    fig, ax = plt.subplots(2,2, dpi=100, figsize=(10,8))
    plt.suptitle(f"Nevents= {Nevents}", size=24)

    # Upper left plot
    logmass_logposterior = logsumexp(log_posterior, axis=0)

    normalisedlogmassposterior = np.exp(logmass_logposterior-logsumexp(logmass_logposterior))

    cdflogmassposterior = np.cumsum(normalisedlogmassposterior)

    mean = logmassrange[np.abs(norm.cdf(0)-cdflogmassposterior).argmin()]

    if credible_levels  is None:
        zscores = [-3, -2,-1,1,2, 3]
        logmasspercentiles = []
        for zscore in zscores:
            logmasspercentiles.append(logmassrange[np.abs(norm.cdf(zscore)-cdflogmassposterior).argmin()])
        
        for o, percentile in enumerate(logmasspercentiles):
            color = colormap(np.abs(zscores[o])/4-0.01)
            
            ax[0,0].axvline(percentile, c=color, ls=':')

        ax[0,0].axvline(mean, c='tab:green', ls=':')
    else:
        logmasspercentiles = []
        for credible_value in credible_levels:
            logmasspercentiles.append(logmassrange[np.abs(credible_value-cdflogmassposterior).argmin()])

        for o, percentile in enumerate(logmasspercentiles):
            color = colormap(o/len(logmasspercentiles))
            
            ax[0,0].axvline(percentile, c=color, ls=':')




    ax[0,0].plot(logmassrange,normalisedlogmassposterior, c='tab:green')



    if truevals!=None:
        ax[0,0].axvline(truevals[1], ls='--', color="tab:orange")

    ax[0,0].set_ylim([0, None])
    ax[0,0].set_xlim([logmassrange[0], logmassrange[-1]])

    # Upper right plot
    ax[0,1].axis('off')


    # Lower left plot
    # ax[1,0].pcolormesh(logmassrange, xi_range, np.exp(normalisedlogposterior).T, cmap='Blues')
    ax[1,0].pcolormesh(logmassrange, xi_range, np.exp(log_posterior), vmin=0)
    if truevals!=None:
        ax[1,0].axvline(truevals[1], c='tab:orange')
        ax[1,0].axhline(truevals[0], c='tab:orange')
    ax[1,0].set_xlabel(r'$log_{10}$ mass [TeV]')
    ax[1,0].set_ylabel(r'$\xi$')

    ax[1,0].set_ylim([xi_range[0], xi_range[-1]])
    ax[1,0].set_xlim([logmassrange[0], logmassrange[-1]])

    ########################################################################################################################
    ########################################################################################################################
    # I have no clue how this works but I've checked it against some standard distributions and it seems correct
    normed_posterior = np.exp(log_posterior)/np.exp(log_posterior).sum()
    n = 100000
    t = np.linspace(0, normed_posterior.max(), n)
    integral = ((normed_posterior >= t[:, None, None]) * normed_posterior).sum(axis=(1,2))

    from scipy import interpolate
    f = interpolate.interp1d(integral, t)
    if credible_levels is None:
        t_contours = f(np.array([1-np.exp(-4.5),1-np.exp(-2.0),1-np.exp(-0.5)]))
    else:
         t_contours = f(np.asarray(credible_levels))
    
    ax[1,0].contour(normed_posterior, t_contours, extent=[logmassrange[0],logmassrange[-1], xi_range[0],xi_range[-1]], colors='white', linewidths=0.5)
    ########################################################################################################################
    ########################################################################################################################


    xi_logposterior = logsumexp(log_posterior, axis=1)

    normalised_xi_posterior = np.exp(xi_logposterior-logsumexp(xi_logposterior))

    cdf_xi_posterior = np.cumsum(normalised_xi_posterior)
    mean_xi = xi_range[np.abs(norm.cdf(0)-cdf_xi_posterior).argmin()]
    xi_percentiles = []
    for zscore in zscores:
        xi_percentile = xi_range[np.abs(norm.cdf(zscore)-cdf_xi_posterior).argmin()]
        xi_percentiles.append(xi_percentile)
        print(np.sqrt(1e5/1e8)*np.abs(xi_percentile - mean_xi))





    ax[1,1].plot(xi_range,normalised_xi_posterior, c='tab:green')

    ax[1,1].axvline(mean_xi, c='tab:green', ls=':')


    for o, percentile in enumerate(xi_percentiles):
                color = colormap(np.abs(zscores[o])/4-0.01)

                ax[1,1].axvline(percentile, c=color, ls=':')
    if truevals!=None:
        ax[1,1].axvline(truevals[0], ls='--', color="tab:orange")
    ax[1,1].set_xlabel(r'$\xi$')
    ax[1,1].set_ylim([0, None])

    if saveplot:
        print(f"Saving to {save_directory}")
        plt.savefig(save_directory)
    plt.show()


# try:
#       runscript = bool(sys.argv[1])
# except:
#       runscript = False

# if runscript:
#     inputs = read_config_file(sys.argv[2])


#     log_posterior   = np.load(f"data/{inputs['identifier']}/log_posterior.npy", allow_pickle=True)
#     xi_range        = np.load(f"data/{inputs['identifier']}/xi_range.npy", allow_pickle=True)
#     logmassrange    = np.load(f"data/{inputs['identifier']}/logmassrange.npy", allow_pickle=True)


#     currentdirecyory = os.getcwd()
#     stemdirectory = f"{currentdirecyory}/data/{inputs['identifier']}/singlerundata"
#     print("\nstem directory: ", stemdirectory, '\n')

#     rundirs = [x[0] for x in os.walk(stemdirectory)][1:]


#     log_posterior = log_posterior-logsumexp(log_posterior)

#     plot_posterior(log_posterior=log_posterior, xi_range=xi_range, logmassrange=logmassrange, 
#                    identifier=inputs['identifier'], Nevents=inputs['Nevents'])