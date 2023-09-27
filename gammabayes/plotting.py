import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import time
from matplotlib import cm
import sys, os
from scipy.special import logsumexp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from gammabayes.utils import log10eaxistrue, longitudeaxistrue, latitudeaxistrue
import os

try:
    stemfolder = f'data/{sys.argv[1]}'
except:
    raise Exception('The identifier you have input is causing an error.')

try:
    profile = f'data/{sys.argv[1]}'
except:
    raise Exception('The identifier you have input is causing an error.')



def convert_lambda_to_sigmav(lambdaval, totalnumevents, log10massDM, 
                             log10eaxis=log10eaxistrue, longitudeaxis=longitudeaxistrue, latitudeaxis=latitudeaxistrue, 
                             logaeff=None, 
                             tobs_seconds=525*60*60, symmetryfactor=1, 
                             fulllogDMmodel=None, fulllogDMspectra=None, DMprofile=None, 
                             mesh=True, singleval = False):
    
    
    aeffunit = aefffull.evaluate(energy_true=np.power(10.,0.0)*u.TeV,
                                                    offset=0*u.deg).to(u.cm**2).unit
    print('DM Diff J Factors...')

    profile = profiles.EinastoProfile()

    # Adopt standard values used in HESS
    profiles.DMProfile.DISTANCE_GC = 8.5 * u.kpc
    profiles.DMProfile.LOCAL_DENSITY = 0.39 * u.Unit("GeV / cm3")

    profile.scale_to_local_density()

    position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
    geom = WcsGeom.create(skydir=position, 
                        binsz=longitudeaxistrue[1]-longitudeaxistrue[0],
                        width=(longitudeaxistrue[-1]-longitudeaxistrue[0]+longitudeaxistrue[1]-longitudeaxistrue[0], latitudeaxistrue[-1]-latitudeaxistrue[0]+longitudeaxistrue[1]-longitudeaxistrue[0]),
                        frame="galactic")


    jfactory = JFactory(
        geom=geom, profile=profile, distance=profiles.DMProfile.DISTANCE_GC
    )
    diffjfactwithunit = jfactory.compute_differential_jfactor().to(u.TeV**2/u.cm**5/u.sr)
    diffjfact = np.log((diffjfactwithunit.value).T)
    
    
    print(f'Units: \naeff --> {aeffunit} \ndiff jfactor --> {diffjfactwithunit.unit} \nDM spectra unit --> 1/TeV')

    
    
    
    truemesh_log10e, truemesh_longitude, truemesh_latitude = np.meshgrid(log10eaxis, longitudeaxis, latitudeaxis, indexing='ij')
    logaeffvals = logaeff(truemesh_log10e.flatten(), np.array([truemesh_longitude.flatten(), truemesh_latitude.flatten()])).reshape(truemesh_log10e.shape)
    
    print('Shapes')
    print(diffjfact.shape)
    print(logaeffvals.shape)
    
    
    print('And now for the spectra values...')
    all_spectral_vals = []
    for logmassval in tqdm(log10massDM):
        spectra_vals = np.squeeze(fulllogDMspectra(logmassval, log10eaxis))
        all_spectral_vals.append(spectra_vals)
    # all_spectral_vals = np.array(all_spectral_vals)
    
    print(all_spectral_vals[0].shape)

    
    if mesh:
        print("And it begins...")
        lambdamesh, logmassmesh = np.meshgrid(lambdaval, log10massDM, indexing='ij')
        sigmavlist = []
        for logmassidx, point in tqdm(enumerate(zip(lambdamesh.flatten(), logmassmesh.flatten())), total=logmassmesh.flatten().shape[0]):
            singlelambdaval, logmassval = point
            sigmav = 4*np.pi*singlelambdaval*totalnumevents/tobs_seconds*(10**(2*logmassval)*2*symmetryfactor)/integrate.simpson(
                integrate.simpson(
                    integrate.simpson(
                        np.exp(all_spectral_vals[np.abs(log10massDM-logmassval).argmin()][:,np.newaxis, np.newaxis]+diffjfact[np.newaxis,:,:]+logaeffvals),
                        10**log10eaxis, axis=0),
                    longitudeaxis, axis=0),
                latitudeaxis)
                
            sigmavlist.append(sigmav)
        sigmavlist = np.array(sigmavlist).reshape(lambdamesh.shape)
        
        return sigmavlist

    else:
        # In this case we presume that we are looking for <sigma v>/signal_fraction^2
        
        sigmav_overxi_list = []
        for logmassval, single_lambda in tqdm(zip(log10massDM, lambdaval), total=log10massDM.shape[0]):
            sigmav_overxi = 4*np.pi*totalnumevents*single_lambda/tobs_seconds*(10**(2*logmassval)*2*symmetryfactor)/integrate.simpson(
                integrate.simpson(
                    integrate.simpson(
                        np.exp(all_spectral_vals[np.abs(log10massDM-logmassval).argmin()][:,np.newaxis, np.newaxis]+diffjfact[np.newaxis,:,:]+logaeffvals),
                        10**log10eaxis, axis=0),
                    longitudeaxis, axis=0),
                latitudeaxis)
                
            sigmav_overxi_list.append(sigmav_overxi)
        sigmav_overxi_list = np.array(sigmav_overxi_list).reshape(log10massDM.shape)
        
        return sigmav_overxi_list

        
    
    


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


if 1:
            
        # Checking contours
        onesigma_contour_vertices = contours.allsegs[2][0]
        onesigma_logmassvals = onesigma_contour_vertices[:,0]
        onesigma_xivals = onesigma_contour_vertices[:,1]
        
        twosigma_contour_vertices = contours.allsegs[1][0]
        twosigma_logmassvals = twosigma_contour_vertices[:,0]
        twosigma_xivals = twosigma_contour_vertices[:,1]
        
        threesigma_contour_vertices = contours.allsegs[0][0]
        threesigma_logmassvals = threesigma_contour_vertices[:,0]
        threesigma_xivals = threesigma_contour_vertices[:,1]
        
        
        
        # sigmavvals = convert_xi_to_sigmav(xivals, totalevents*eventmultiplier, logmassvals, log10eaxistrue, longitudeaxistrue, latitudeaxistrue, aeff, 52.5*60*60, 1, fulllogDMmodel=None, fulllogDMspectra=None, DMdiffJfactor_func=None)
        
            
        
        plt.figure()
        plt.pcolormesh(logmassrange, xirange, np.exp(log_posterior))
        plt.plot(onesigma_logmassvals, onesigma_xivals, c='white')
        plt.plot(twosigma_logmassvals, twosigma_xivals, c='red')
        plt.plot(threesigma_logmassvals, threesigma_xivals, c='yellow')
        plt.show()
        
        sigmavvals = convert_xi_to_sigmav(twosigma_xivals, totalevents*eventmultiplier, twosigma_logmassvals, log10eaxistrue, 
                                                longitudeaxistrue, latitudeaxistrue, aefffunc, 52.5*60*60, 1, fulllogDMmodel=None, fulllogDMspectra=energymassinputspectralfunc, mesh=False)

        window_size = 10
        cubicfit = np.polyfit(twosigma_logmassvals, np.log(sigmavvals/np.sqrt(10)), 3)

        
        def cubic_func(x, polyfitoutput):
            p0, p1, p2, p3 = polyfitoutput[3], polyfitoutput[2], polyfitoutput[1], polyfitoutput[0]
            return p0 + x*p1 + x**2*p2 + x**3*p3
                
        # sorted_indices = np.argsort(twosigma_logmassvals)
        import csv
        
        aacharya = []
        with open("sensitivitypaper_points.csv", 'r', encoding='UTF8') as file:
            csv_reader = csv.reader(file, delimiter=',')
            for row in csv_reader:
                aacharya.append([float(row[0]),float(row[1])])

        aacharya = np.array(aacharya)
        
        
        
        plt.figure(figsize=(4,3), dpi=200)
        # plt.plot(10**twosigma_logmassvals, sigmavvals/np.sqrt(10))
        # plt.plot(10**twosigma_logmassvals[window_size:-window_size], np.exp(sigma_smoothed)[window_size:-window_size])
        plt.plot(10**twosigma_logmassvals[window_size:], np.exp(cubic_func(twosigma_logmassvals[window_size:], cubicfit)), label='this work')
        plt.plot(aacharya[:,0]/1e3, aacharya[:,1], label='CTA consortium 2021')
        # plt.axhline(2e-26, ls='--', c='grey', alpha=0.5, label=r'DarkSUSY thermal $\langle \sigma v \rangle$')
        plt.ylim([1e-27, 1e-24])
        plt.xlim([1e-1,1e2])
        plt.grid(color='grey', alpha=0.2)
        plt.loglog()
        plt.xlabel(r'$m_\chi$ [TeV]')
        plt.ylabel(r'$\langle \sigma v \rangle$ [cm$^3$ s$^{-1}$]')
        plt.text(1.1e-1, 1.1e-27, 'signal: Einasto \nbackground: CR + IEM', fontsize=6)
        plt.legend(fontsize=6)
        plt.tight_layout()
        # plt.savefig('Figures/sensitivity_plot_12_09_23.png')
        plt.show()