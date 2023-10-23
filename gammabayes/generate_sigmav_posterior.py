import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import integrate, special
from utils.utils import aeff_efficient, convertlonlat_to_offset
from utils.event_axes import log10eaxistrue, longitudeaxistrue, latitudeaxistrue
from gammapy.astro.darkmatter import (
    profiles,
    JFactory
)
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
from tqdm import tqdm
import os, sys
from inverse_transform_sampling import inverse_transform_sampling
from SS_DM_Prior import SS_DM_dist
from os import path
from utils.config_utils import read_config_file


aux_data_dir = path.join(path.dirname(__file__), 'aux_data')



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
logdiffjfact = np.log((diffjfactwithunit.value).T)



def convert_to_sigmav(xi_val, log10massDM, totalnumevents, 
                             log10eaxis=log10eaxistrue, longitudeaxis=longitudeaxistrue, latitudeaxis=latitudeaxistrue, 
                             logaeff=aeff_efficient, 
                             tobs_seconds=525*60*60, symmetryfactor=1, 
                             fulllogDMspectra=None, logdiffjfact=logdiffjfact):
    

    truemesh_log10e, truemesh_longitude, truemesh_latitude = np.meshgrid(log10eaxis, longitudeaxis, latitudeaxis, indexing='ij')
    logaeffvals = logaeff(truemesh_log10e.flatten(), 
                          convertlonlat_to_offset(np.array([truemesh_longitude.flatten(), truemesh_latitude.flatten()]))).reshape(
                              truemesh_log10e.shape)

  
    spectramesh_logmass, spectramesh_logenergy = np.meshgrid(log10massDM, log10eaxis, indexing='ij')
    spectra_vals = np.squeeze(fulllogDMspectra(spectramesh_logmass.flatten(), spectramesh_logenergy.flatten()).reshape(spectramesh_logmass.shape))
        
    lonlat_integral = np.log(integrate.simpson(integrate.simpson(
                np.exp(logdiffjfact[np.newaxis, :,:]+logaeffvals),
                longitudeaxis, axis=1),
            latitudeaxis, axis=1))
    
    
    sigmav = 4*np.pi*xi_val*totalnumevents/tobs_seconds*(10**(2*log10massDM)*2*symmetryfactor)/integrate.simpson(
        np.exp(spectra_vals+lonlat_integral[np.newaxis, :]),
        10**log10eaxis, axis=1)
    
    print(sigmav.shape)

    return sigmav



try:
    stemfolder = f'data/{sys.argv[1]}'
except:
    raise Exception('The identifier you have input is causing an error.')

try:
    totalevents = int(float(sys.argv[2]))
except:
    raise Exception('Number of events not input cannot calculate sigma v')


try:
    sensitivity_plot = int(sys.argv[3])
except:
    sensitivity_plot = False
    
try:
    projectednumber = int(sys.argv[4])
except:
    projectednumber = int(float(1e8))


log_posterior   = np.load(f'{stemfolder}/log_posterior.npy', allow_pickle=True)
xi_range        = np.load(f'{stemfolder}/xi_range.npy', allow_pickle=True)
logmassrange    = np.load(f'{stemfolder}/logmassrange.npy', allow_pickle=True)


log_posterior               = log_posterior-special.logsumexp(log_posterior)
sampledindices              = inverse_transform_sampling(log_posterior.flatten(), Nsamples=int(1e4))
reshaped_sampled_indices    = np.unravel_index(sampledindices,log_posterior.shape)
logmass_values              = logmassrange[reshaped_sampled_indices[1]]
xi_values                   = xi_range[reshaped_sampled_indices[0]]



params = read_config_file(f"data/{sys.argv[1]}/singlerundata/inputconfig.yaml")


true_xi         = params['xi']
truelogmass     = params['logmass']



SS_DM_Class_instance = SS_DM_dist(longitudeaxistrue, latitudeaxistrue, density_profile=profiles.EinastoProfile(), ratios=True)



sigmav_samples = convert_to_sigmav(xi_val=xi_values, log10massDM=logmass_values, totalnumevents=totalevents, 
                                   fulllogDMspectra=SS_DM_Class_instance.nontrivial_coupling, tobs_seconds=52.5*60*60,
                                   )/np.sqrt(projectednumber/totalevents)

truesigmav = convert_to_sigmav(xi_val=true_xi, log10massDM=truelogmass, totalnumevents=totalevents, 
                                   fulllogDMspectra=SS_DM_Class_instance.nontrivial_coupling, tobs_seconds=52.5*60*60,
                                   )/np.sqrt(projectednumber/totalevents)
print('\n\n')
print(sigmav_samples)
print('\n\n')

# Now you have samples of sigmav. You can use them to estimate its PDF.
# For example, you can use a histogram to estimate the PDF.

# Create a histogram to estimate the PDF of sigmav
plt.figure(dpi=300, figsize=(5,4))
upperbound = np.quantile(sigmav_samples, 0.997)

lowerbound = np.quantile(sigmav_samples, 0.003)

if lowerbound==0:
    lowerbound = np.min(sigmav_samples[sigmav_samples>0])

print(f"Upper: {upperbound}\nLower: {lowerbound}")
# hist, bins, patches = plt.hist(sigmav_samples, density=True, bins=np.logspace(np.log10(lowerbound), np.log10(upperbound), 61))
hist, bins, patches = plt.hist(sigmav_samples, density=True, bins=np.linspace(lowerbound, upperbound, 61))
plt.axvline(truesigmav, c="tab:orange", ls='--')

# plt.axvline(np.quantile(sigmav_samples, 0.90), c='tab:orange')
# plt.axvline(np.quantile(sigmav_samples, 0.10), c='tab:orange')


plt.xlabel(r'$\langle \sigma v \rangle$', size=14)
plt.ylim([0,None])
# plt.xscale('log')
# plt.xlim(np.quantile(sigmav_samples, 0.04), np.quantile(sigmav_samples, 0.96))
plt.tight_layout()
# plt.savefig(f"data/{sys.argv[1]}/sigmav_hist.pdf")
plt.savefig(f"data/{sys.argv[1]}/sigmav_hist_linear_binning.pdf")
plt.show()



if sensitivity_plot:
    fig, ax = plt.subplots()
    normed_posterior = np.exp(log_posterior-special.logsumexp(log_posterior))
    
    ax.pcolormesh(logmassrange, xi_range, normed_posterior)

    n = 10000
    t = np.linspace(0, normed_posterior.max(), n)
    integral = ((normed_posterior >= t[:, None, None]) * normed_posterior).sum(axis=(1,2))

    from scipy import interpolate
    f = interpolate.interp1d(integral, t)
    t_contours = f(np.array([1-np.exp(-4.5),1-np.exp(-2.0),1-np.exp(-0.5), 0.05]))
    contours = ax.contour(normed_posterior, t_contours, extent=[logmassrange[0],logmassrange[-1], xi_range[0],xi_range[-1]], colors='white', linewidths=0.5)
    plt.show()
    
    
    
    # Checking contours
    onesigma_contour_vertices = contours.allsegs[2][0]
    onesigma_logmassvals = onesigma_contour_vertices[:,0]
    onesigma_xi_vals = onesigma_contour_vertices[:,1]
    
    twosigma_contour_vertices = contours.allsegs[1][0]
    twosigma_logmassvals = twosigma_contour_vertices[:,0]
    twosigma_xi_vals = twosigma_contour_vertices[:,1]
    
    threesigma_contour_vertices = contours.allsegs[0][0]
    threesigma_logmassvals = threesigma_contour_vertices[:,0]
    threesigma_xi_vals = threesigma_contour_vertices[:,1]
    
    plt.figure()
    plt.pcolormesh(logmassrange, xi_range, normed_posterior)
    plt.plot(onesigma_logmassvals, onesigma_xi_vals, c='white')
    plt.plot(twosigma_logmassvals, twosigma_xi_vals, c='red')
    plt.plot(threesigma_logmassvals, threesigma_xi_vals, c='yellow')
    plt.show()
    

    sigmavvals = convert_to_sigmav(xi_val=twosigma_xi_vals, log10massDM=twosigma_logmassvals, totalnumevents=totalevents, 
                                   fulllogDMspectra=SS_DM_Class_instance.nontrivial_coupling, tobs_seconds=52.5*60*60,
                                   )/np.sqrt(projectednumber/totalevents)

            
    # sorted_indices = np.argsort(twosigma_logmassvals)
    import csv
    
    aacharya = []
    with open(f"{aux_data_dir}/sensitivitypaper_points.csv", 'r', encoding='UTF8') as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            aacharya.append([float(row[0]),float(row[1])])

    aacharya = np.array(aacharya)

    
    plt.figure(figsize=(4,3), dpi=200)
    # plt.plot(10**twosigma_logmassvals, sigmavvals/np.sqrt(10))
    # plt.plot(10**twosigma_logmassvals[window_size:-window_size], np.exp(sigma_smoothed)[window_size:-window_size])
    # plt.plot(10**twosigma_logmassvals[window_size:], np.exp(cubic_func(twosigma_logmassvals[window_size:], cubicfit)), label='this work')
    plt.plot(10**twosigma_logmassvals, sigmavvals, label='this work')
    plt.plot(aacharya[:,0]/1e3, aacharya[:,1], label='CTA consortium 2021')
    # plt.axhline(2e-26, ls='--', c='grey', alpha=0.5, label=r'DarkSUSY thermal $\langle \sigma v \rangle$')
    plt.ylim([1e-27, 1e-23])
    plt.xlim([1e-1,1e2])
    plt.grid(color='grey', alpha=0.2)
    plt.loglog()
    plt.xlabel(r'$m_\chi$ [TeV]')
    plt.ylabel(r'$\langle \sigma v \rangle$ [cm$^3$ s$^{-1}$]')
    plt.text(1.1e-1, 1.1e-27, 'signal: Einasto \nbackground: CR + IEM', fontsize=6)
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"data/{sys.argv[1]}/sensitivity_plot.pdf")
    plt.show()
            
            