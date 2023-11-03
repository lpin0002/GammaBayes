import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import time
from matplotlib import cm
import sys, os
from scipy.special import logsumexp
from .utils import hdp_credible_interval_1d, bin_centres_to_edges
from .event_axes import log10eaxistrue, longitudeaxistrue, latitudeaxistrue, log10eaxis, longitudeaxis, latitudeaxis
from matplotlib.colors import LogNorm

def logdensity_matrix_plot(axes, logprobmatrix, truevals=None, sigmalines_1d=True, sigmas=range(0,6), 
                           cmap=cm.get_cmap('Blues_r'), 
                           levels=np.array([1-np.exp(-25/2), 1-np.exp(-8), 1-np.exp(-4.5),1-np.exp(-2.0),1-np.exp(-0.5)]),
                           axis_names=None, suptitle='', suptitlesize=12,
                           **kwargs):
    numaxes = len(axes)
    n = 1000

    fig, ax = plt.subplots(numaxes, numaxes, **kwargs)
    plt.suptitle(suptitle, size=suptitlesize)
    for rowidx in range(numaxes):
        for colidx in range(numaxes):
            if rowidx==colidx:
                marginal_prob = np.exp(logsumexp(logprobmatrix, axis=(*np.delete(np.arange(numaxes), [rowidx]),)))
                marginal_prob = marginal_prob/simps(y=marginal_prob, x=axes[rowidx])
                ax[rowidx,rowidx].plot(axes[rowidx], marginal_prob)
                if truevals!=None:
                    ax[rowidx, rowidx].axvline(truevals[rowidx], ls='--', lw=1, c='tab:orange')
                ax[rowidx,rowidx].set_ylim([0,None])
                ax[rowidx, rowidx].set_xlim([axes[colidx].min(), axes[colidx].max()])



                if sigmalines_1d:
                    for sigma in sigmas:
                        if sigma!=0:
                            color = cmap((np.abs(sigma)+0.1)/(2*len(sigmas)))
                        else:
                            color = 'tab:green'
                        for val in hdp_credible_interval_1d(y=marginal_prob, x=axes[rowidx], sigma=sigma):
                            ax[rowidx, rowidx].axvline(val, ls='--',c=color, alpha=0.8)


                if axis_names is not None:
                    ax[rowidx, rowidx].set_xlabel(axis_names[rowidx])
            elif colidx<rowidx:
                integrationaxes = np.delete(np.arange(numaxes), [rowidx,colidx])
                marginal_prob = np.exp(logsumexp(logprobmatrix, axis=(*integrationaxes,))).T

                ax[rowidx, colidx].pcolormesh(axes[colidx], axes[rowidx], marginal_prob)
                if truevals!=None:
                    ax[rowidx,colidx].axvline(truevals[colidx], ls='--', lw=1, c='tab:orange')
                    ax[rowidx,colidx].axhline(truevals[rowidx], ls='--', lw=1, c='tab:orange')

                ax[rowidx,colidx].set_xlim([axes[colidx].min(), axes[colidx].max()])
                ax[rowidx,colidx].set_ylim([axes[rowidx].min(), axes[rowidx].max()])

                normed_marginal_prob = (marginal_prob/marginal_prob.sum())
                t = np.linspace(0, normed_marginal_prob.max(), n)
                integral = ((normed_marginal_prob >= t[:, None, None]) * normed_marginal_prob).sum(axis=(1,2))

                f = interp1d(integral, t)
                t_contours = f(levels)
                ax[rowidx,colidx].contour(normed_marginal_prob, t_contours, 
                                extent=[axes[colidx].min(), axes[colidx].max(), axes[rowidx].min(), axes[rowidx].max(),], colors='white', linewidths=0.5)
                if axis_names is not None:
                    ax[rowidx, colidx].set_xlabel(axis_names[colidx])
                    ax[rowidx, colidx].set_ylabel(axis_names[rowidx])
            else:
                ax[rowidx,colidx].set_axis_off()
                
    plt.tight_layout()


def plot_loge_lon_lat_prior_samples(samples, 
    twodhist_xlabel='Galactic Longitude [deg]', twodhist_ylabel='Galactic Latitude [deg]', 
    onedhist_xlabel = r'Log$_{10}$ Energy [TeV]',
    onedhist_bins = bin_centres_to_edges(log10eaxistrue),
    twodhist_bins = (bin_centres_to_edges(longitudeaxistrue), bin_centres_to_edges(latitudeaxistrue)),
    dist_names = None, 
    log10eaxistrue=log10eaxistrue, longitudeaxistrue=longitudeaxistrue, latitudeaxistrue=latitudeaxistrue, 
    **kwargs,):

    num_priors = len(samples)

    fig, ax = plt.subplots(num_priors,2, **kwargs)
    for idx, prior_samples in enumerate(samples):
        if dist_names!=None:
            ax[idx,0].set_title(dist_names[idx])
        ax[idx,0].set_xlabel(twodhist_xlabel)
        ax[idx,0].set_ylabel(twodhist_ylabel)
        hist2dplot = ax[idx,0].hist2d(prior_samples[1],prior_samples[2], 
            bins=twodhist_bins, norm=LogNorm())
        plt.colorbar(mappable=hist2dplot[3])

        ax[idx,1].hist(prior_samples[0], bins=onedhist_bins)
        ax[idx,1].set_xlabel(onedhist_xlabel)

    plt.tight_layout()


def plot_loge_lon_lat_recon_samples(samples, 
    twodhist_xlabel='Galactic Longitude [deg]', twodhist_ylabel='Galactic Latitude [deg]', 
    onedhist_xlabel = r'Log$_{10}$ Energy [TeV]',
    onedhist_bins = bin_centres_to_edges(log10eaxis),
    twodhist_bins = (bin_centres_to_edges(longitudeaxis), bin_centres_to_edges(latitudeaxis)),
    dist_names = None, 
    log10eaxis=log10eaxis, longitudeaxis=longitudeaxis, latitudeaxis=latitudeaxis, 
    **kwargs,):

    num_priors = len(samples)

    fig, ax = plt.subplots(num_priors,2, **kwargs)
    for idx, recon_samples in enumerate(samples):
        if dist_names!=None:
            ax[idx,0].set_title(dist_names[idx])
        ax[idx,0].set_xlabel(twodhist_xlabel)
        ax[idx,0].set_ylabel(twodhist_ylabel)
        hist2dplot = ax[idx,0].hist2d(recon_samples[1],recon_samples[2], 
            bins=twodhist_bins, norm=LogNorm())
        plt.colorbar(mappable=hist2dplot[3])

        ax[idx,1].hist(recon_samples[0], bins=onedhist_bins)
        ax[idx,1].set_xlabel(onedhist_xlabel)

    plt.tight_layout()

    