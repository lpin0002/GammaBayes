import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import time
from matplotlib import cm
import sys, os
from scipy.special import logsumexp
from gammabayes.utils import hdp_credible_interval_1d, bin_centres_to_edges, iterate_logspace_simps, logspace_simpson
from gammabayes.utils.event_axes import energy_true_axis, longitudeaxistrue, latitudeaxistrue, energy_recon_axis, longitudeaxis, latitudeaxis
from matplotlib.colors import LogNorm



defaults_kwargs = dict(
    smooth=1.0, 
    label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16), color='#0072C1',
    truth_color='tab:orange', 
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.), 1 - np.exp(-8), 1 - np.exp(-25/2)),
    plot_density=True, 
    plot_datapoints=True, 
    fill_contours=True,
    max_n_ticks=4, 
    )



def logdensity_matrix_plot(axes, log_dist_matrix, truevals=None, sigmalines_1d=True, sigmas=range(0,6), 
                           cmap=cm.get_cmap('Blues_r'), contours2d = False,
                           levels=np.array([1-np.exp(-25/2), 1-np.exp(-8), 1-np.exp(-4.5),1-np.exp(-2.0),1-np.exp(-0.5)]),
                           axis_names=None, suptitle='', suptitlesize=12, plot_density=False, norm=None,
                           single_dim_yscales='symlog', single_dim_ylabel='', vmin=None, vmax=None,
                           **kwargs):
    numaxes = len(axes)
    n = 1000

    fig, ax = plt.subplots(numaxes, numaxes, **kwargs)
    plt.suptitle(suptitle, size=suptitlesize)
    for rowidx in range(numaxes):
        for colidx in range(numaxes):
            if rowidx==colidx:
                integration_axes_indices = np.delete(np.arange(numaxes), [rowidx])
                integration_axes = [axes[integration_axis_idx] for integration_axis_idx in integration_axes_indices]
                log_marginal_dist = iterate_logspace_simps(log_dist_matrix, 
                                        axes=integration_axes, 
                                        axisindices=integration_axes_indices)
                if plot_density:
                    marginal_dist = np.exp(log_marginal_dist-logspace_simpson(logy=log_marginal_dist, x=axes[rowidx]))
                else:
                    marginal_dist = np.exp(log_marginal_dist)
                ax[rowidx,rowidx].plot(axes[rowidx], marginal_dist)
                if not(truevals is None):
                    ax[rowidx, rowidx].axvline(truevals[rowidx], ls='--', lw=1, c='tab:orange')
                ax[rowidx,rowidx].set_ylim([0,None])
                ax[rowidx, rowidx].set_xlim([axes[colidx].min(), axes[colidx].max()])
                ax[rowidx,rowidx].set_yscale(single_dim_yscales)



                if sigmalines_1d and plot_density:
                    print(marginal_dist.shape, axes[rowidx].shape)
                    for sigma in sigmas:
                        if sigma!=0:
                            color = cmap((np.abs(sigma)+0.1)/(2*len(sigmas)))
                        else:
                            color = 'tab:green'
                        for val in hdp_credible_interval_1d(y=marginal_dist, x=axes[rowidx], sigma=sigma):
                            ax[rowidx, rowidx].axvline(val, ls=':',c=color, alpha=0.8)

                if rowidx==0:
                    ax[rowidx, rowidx].set_ylabel(single_dim_ylabel)


                if not(axis_names is None) and rowidx==numaxes-1:
                    ax[rowidx, rowidx].set_xlabel(axis_names[rowidx])

            elif colidx<rowidx:
                integration_axes_indices = np.delete(np.arange(numaxes), [rowidx,colidx])
                integration_axes = [axes[integration_axis_idx] for integration_axis_idx in integration_axes_indices]
                log_marginal_dist = iterate_logspace_simps(log_dist_matrix, 
                                                axes=integration_axes, 
                                                axisindices=integration_axes_indices).T

                ax[rowidx, colidx].pcolormesh(axes[colidx], axes[rowidx], np.exp(log_marginal_dist), norm=norm, vmin=vmin, vmax=vmax)
                if truevals!=None:
                    ax[rowidx,colidx].axvline(truevals[colidx], ls='--', lw=1, c='tab:orange')
                    ax[rowidx,colidx].axhline(truevals[rowidx], ls='--', lw=1, c='tab:orange')

                ax[rowidx,colidx].set_xlim([axes[colidx].min(), axes[colidx].max()])
                ax[rowidx,colidx].set_ylim([axes[rowidx].min(), axes[rowidx].max()])
                if plot_density and contours2d:
                    normed_marginal_dist = np.exp(log_marginal_dist - logsumexp(log_marginal_dist))
                    t = np.linspace(0, normed_marginal_dist.max(), n)
                    integral = ((normed_marginal_dist >= t[:, None, None]) * normed_marginal_dist).sum(axis=(1,2))

                    f = interp1d(integral, t)
                    t_contours = f(levels)
                    ax[rowidx,colidx].contour(axes[colidx], axes[rowidx], normed_marginal_dist, levels=t_contours, colors='white', linewidths=0.5)
    
                if not(axis_names is None) and rowidx==numaxes-1:
                    ax[rowidx, colidx].set_xlabel(axis_names[colidx])
                if not(axis_names is None) and colidx==0:
                    ax[rowidx, colidx].set_ylabel(axis_names[rowidx])
            else:
                ax[rowidx,colidx].set_axis_off()

    plt.tight_layout()

    return fig,ax
                


def plot_loge_lon_lat_prior_samples(samples, 
    twodhist_xlabel='Galactic Longitude [deg]', twodhist_ylabel='Galactic Latitude [deg]', 
    onedhist_xlabel = r'Log$_{10}$ Energy [TeV]',
    onedhist_bins = bin_centres_to_edges(energy_true_axis),
    twodhist_bins = (bin_centres_to_edges(longitudeaxistrue), bin_centres_to_edges(latitudeaxistrue)),
    dist_names = None, 
    energy_true_axis=energy_true_axis, longitudeaxistrue=longitudeaxistrue, latitudeaxistrue=latitudeaxistrue, 
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
    onedhist_bins = bin_centres_to_edges(energy_recon_axis),
    twodhist_bins = (bin_centres_to_edges(longitudeaxis), bin_centres_to_edges(latitudeaxis)),
    dist_names = None, 
    log10eaxis=energy_recon_axis, longitudeaxis=longitudeaxis, latitudeaxis=latitudeaxis, 
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

    