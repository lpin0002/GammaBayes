import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import time
from matplotlib import cm
import sys
from scipy.special import logsumexp
from gammabayes.utils import log10eaxistrue, longitudeaxistrue, latitudeaxistrue
import os

try:
    stemfolder = f'data/{sys.argv[1]}'
except:
    raise Exception('The identifier you have input is causing an error.')

log_posterior   = np.load(f'{stemfolder}/log_posterior.npy', allow_pickle=True)
xi_range        = np.load(f'{stemfolder}/xi_range.npy', allow_pickle=True)
logmassrange    = np.load(f'{stemfolder}/logmassrange.npy', allow_pickle=True)
params         = np.load(f'{stemfolder}/singlerundata/1/params.npy', allow_pickle=True).item()


currentdirecyory = os.getcwd()
stemdirectory = f'{currentdirecyory}/{stemfolder}/singlerundata'
print("\nstem directory: ", stemdirectory, '\n')

rundirs = [x[0] for x in os.walk(stemdirectory)][1:]



Nevents = params['Nevents']*len(rundirs)
true_xi = params['true_xi']
truelogmass = params['true_log10_mass']



log_posterior = log_posterior-logsumexp(log_posterior)

colormap = cm.get_cmap('Blues_r', 4)

fig, ax = plt.subplots(2,2, dpi=100, figsize=(10,8))
plt.suptitle(f"Nevents= {Nevents}", size=24)

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
ax[0,0].axvline(truelogmass, ls='--', color="tab:orange")


if min(mean - logmasspercentiles)>log10eaxistrue[1]-log10eaxistrue[0]:
    for logetrueval in log10eaxistrue:
        ax[0,0].axvline(logetrueval, c='forestgreen', alpha=0.3)
ax[0,0].set_ylim([0, None])
ax[0,0].set_xlim([logmassrange[0], logmassrange[-1]])

# Upper right plot
ax[0,1].axis('off')


# Lower left plot
# ax[1,0].pcolormesh(logmassrange, xi_range, np.exp(normalisedlogposterior).T, cmap='Blues')
ax[1,0].pcolormesh(logmassrange, xi_range, np.exp(log_posterior), vmin=0)
ax[1,0].axvline(truelogmass, c='tab:orange')
ax[1,0].axhline(true_xi, c='tab:orange')
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
ax[1,1].axvline(true_xi, ls='--', color="tab:orange")
ax[1,1].set_xlabel(r'$\xi$')
ax[1,1].set_ylim([0, None])


plt.savefig(time.strftime(f"{stemfolder}/posteriorplot_%m%d_%H.pdf"))
plt.show()