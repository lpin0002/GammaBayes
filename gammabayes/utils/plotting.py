import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib import cm
from scipy.special import logsumexp
import os, time, numpy as np


def mixture_posterior_plot(log_posterior, xi_range, logmassrange, config_file=None, saveplot=True, savepath=''):
    """Plots the input log_posterior over input hyperparameter axes

    Args:
        log_posterior (np.ndarray): The log posterior values to be plotted

        xi_range (np.ndarray): Values of the mixture weights to be used.

        logmassrange (np.ndarray): Values of logmass to be used during plotting

        config_file (dict, optional): Configuration parameter to be used to add 
        information to the plot. Defaults to None.

        saveplot (bool, optional): A bool of whether to save the plot if config 
        file given it uses the identifier for where to save the plot. 
        Defaults to True.

        savepath (str, optional): If config file not given then this is used for
        the file path to save the plot. If both this and the config file are not
        given and saveplot==True, then the plot is saved to the working 
        directory. Defaults to '' (working directory).
        
    """
    currentdirecyory = os.getcwd()
    if not(config_file is None):
        stemdirectory = f"{currentdirecyory}/data/{config_file['identifier']}/singlerundata"
        print("\nstem directory: ", stemdirectory, '\n')

    log_posterior = log_posterior-logsumexp(log_posterior)

    colormap = cm.get_cmap('Blues_r', 4)

    fig, ax = plt.subplots(2,2, dpi=100, figsize=(10,8))

    if not(config_file is None):
        plt.suptitle(f"Nevents= {config_file['totalevents']}", size=24)

    # Upper left plot
    logmass_logposterior = logsumexp(log_posterior, axis=0)

    normalisedlogmassposterior = np.exp(logmass_logposterior-logsumexp(logmass_logposterior))

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

    if not(config_file is None):
        ax[0,0].axvline(config_file['logmass'], ls='--', color="tab:orange")

    ax[0,0].set_ylim([0, None])
    ax[0,0].set_xlim([logmassrange[0], logmassrange[-1]])

    # Upper right plot
    ax[0,1].axis('off')


    # Lower left plot
    # ax[1,0].pcolormesh(logmassrange, xi_range, np.exp(normalisedlogposterior).T, cmap='Blues')
    ax[1,0].pcolormesh(logmassrange, xi_range, np.exp(log_posterior), vmin=0)
    if not(config_file is None):
        ax[1,0].axvline(config_file['logmass'], c='tab:orange')
        ax[1,0].axhline(config_file['xi'], c='tab:orange')

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
    t_contours = f(np.array([1-np.exp(-4.5),1-np.exp(-2.0),1-np.exp(-0.5)]))
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
    if not(config_file is None):
          ax[1,1].axvline(config_file['xi'], ls='--', color="tab:orange")
    ax[1,1].set_xlabel(r'$\xi$')
    ax[1,1].set_ylim([0, None])

    if (config_file is None) and saveplot:
        plt.savefig(time.strftime(f"{savepath}/posteriorplot_%m%d_%H.pdf"))
    elif saveplot:
        plt.savefig(time.strftime(f"data/{config_file['identifier']}/posteriorplot_%m%d_%H.pdf"))

    plt.show()


