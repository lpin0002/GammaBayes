from utils3d import *
from BFCalc.createspectragrids import darkmatterdoubleinput, energymassinputspectralfunc
from scipy import special
import os, time, sys, numpy as np, matplotlib.pyplot as plt, corner.corner as corner
from matplotlib import cm
import matplotlib as mpl
from scipy.stats import norm
from tqdm import tqdm
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

try:
    identifier = sys.argv[1]
except:
    identifier = time.strftime("%d%m%H")
try:
    showhyperparameterposterior = int(sys.argv[2])
except:
    showhyperparameterposterior = 1
try:
    shownuisanceparameterposterior = int(sys.argv[3])
except:
    shownuisanceparameterposterior = 0
try:
    showsamples = int(sys.argv[4])
except:
    showsamples = 0
    
try: 
    shownumberofsamples = int(sys.argv[5])
except:
    shownumberofsamples = 0

integrationtype = '_direct'
    
specsetup = darkmatterdoubleinput


from gammapy.astro.darkmatter import (
    profiles,
    JFactory
)



def convert_lambda_to_sigmav(lambdaval, totalnumevents, log10massDM, log10eaxis=log10eaxistrue, longitudeaxis=longitudeaxistrue, latitudeaxis=latitudeaxistrue, logaeff=None, tobs_seconds=525*60*60, symmetryfactor=1, fulllogDMmodel=None, fulllogDMspectra=None, DMprofile=None):
    
    
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

    
    print('And now for the spectra values...')
    all_spectral_vals = []
    for logmassval in tqdm(log10massDM):
        spectra_vals = np.squeeze(fulllogDMspectra(logmassval, log10eaxis))
        all_spectral_vals.append(spectra_vals)
    # all_spectral_vals = np.array(all_spectral_vals)
    
    truemesh_log10e, truemesh_longitude, truemesh_latitude = np.meshgrid(log10eaxis, longitudeaxis, latitudeaxis, indexing='ij')
    logaeffvals = logaeff(truemesh_log10e.flatten(), np.array([truemesh_longitude.flatten(), truemesh_latitude.flatten()])).reshape(truemesh_log10e.shape)
    
    print('Shapes')
    print(diffjfact.shape)
    print(logaeffvals.shape)
    print(all_spectral_vals[0].shape)
    
    print("And it begins...")
    sigmavlist = []
    for logmassidx, point in tqdm(enumerate(zip(log10massDM,lambdaval)), total=log10massDM.shape[0]):
        logmassval, singlelambdaval = point
        sigmav = 4*np.pi*singlelambdaval*totalnumevents/tobs_seconds*(10**(2*logmassval)*2*symmetryfactor)/integrate.simpson(
            integrate.simpson(
                integrate.simpson(
                    np.exp(all_spectral_vals[logmassidx][:,np.newaxis, np.newaxis]+diffjfact[np.newaxis,:,:]+logaeffvals),
                    10**log10eaxis, axis=0),
                longitudeaxis, axis=0),
            latitudeaxis)
            
        sigmavlist.append(sigmav)
    
    
    return sigmavlist
    



whattoplot = [showhyperparameterposterior,shownuisanceparameterposterior,showsamples]

currentdirecyory = os.getcwd()
stemdirectory = currentdirecyory+f'/data/{identifier}'
print("\nstem directory: ", stemdirectory, '\n')

rundirs = [x[0] for x in os.walk(stemdirectory)][1:]
print("number of run directories: ", len(rundirs), '\n')



if True:
    params                  = np.load(f"{rundirs[0]}/params.npy")
    totalevents             = int(params[1,1])
    truelambda              = float(params[1,0])
    truelogmass             = float(params[1,2])
    
    if showhyperparameterposterior:
        logmassrange = np.load(f'{stemdirectory}/logmassrange{integrationtype}.npy')
        lambdarange = np.load(f'{stemdirectory}/lambdarange{integrationtype}.npy')

    # lambdarange = np.load(f'{rundirs[0]}/lambdarange{integrationtype}.npy')

    for rundir in rundirs[1:]:
            runnum = rundir.replace(stemdirectory+'/', '')
            print("runnum: ", runnum)
            params              = np.load(f"data/{identifier}/{runnum}/params.npy")
            
            # truesigsamples       =np.concatenate((truesigsamples, np.load(f"{rundir}/truesigsamples.npy")))
            # truebkgsamples       =np.concatenate((truebkgsamples, np.load(f"{rundir}/truebkgsamples.npy")))
            # meassigsamples       =np.concatenate((meassigsamples, np.load(f"{rundir}/meassigsamples.npy")))
            # measbkgsamples       =np.concatenate((measbkgsamples, np.load(f"{rundir}/measbkgsamples.npy")))
            truelogmass     = float(params[1,2])
            nevents         = int(params[1,1])
            totalevents+=nevents
            truelambdaval   = float(params[1,0])

    if showhyperparameterposterior:
        
        import matplotlib.colors as colors
        lambdafraction = 1.0
        logmassfraction = 1.0
        eventmultiplier = 50
        
        
        
        unnormalised_log_posterior = np.load(f'{stemdirectory}/unnormalised_logposterior_direct.npy')
        log_posterior = unnormalised_log_posterior-special.logsumexp(unnormalised_log_posterior)
        
        
        from utils3d import confidence_ellipse
        from scipy.stats import norm

        import time

        colormap = cm.get_cmap('Blues_r', 4)

        fig, ax = plt.subplots(2,2, dpi=100, figsize=(10,8))
        plt.suptitle(f"Nevents= {totalevents*eventmultiplier}", size=24)

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
        ax[0,0].axvline(truelogmass, ls='--', color="tab:orange")


        
        # for logetrueval in log10eaxis:
        #     ax[0,0].axvline(logetrueval, c='forestgreen', alpha=0.3)
        ax[0,0].set_ylim([0, None])
        ax[0,0].set_xlim([logmassrange[0], logmassrange[-1]])
        ax[0,0].ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)


        # Upper right plot
        ax[0,1].axis('off')


        # Lower left plot
        # ax[1,0].pcolormesh(logmassrange, lambdarange, np.exp(normalisedlogposterior).T, cmap='Blues')
        pcol = ax[1,0].pcolormesh(logmassrange, lambdarange, np.exp(log_posterior))
        pcol.set_edgecolor('face')
        ax[1,0].grid(False)
        ax[1,0].axvline(truelogmass, c='tab:orange')
        ax[1,0].axhline(truelambda, c='tab:orange')
        ax[1,0].set_xlabel(r'$log_{10} (m_\chi)$ [TeV]')
        ax[1,0].set_ylabel(r'$\lambda$')

        ax[1,0].set_ylim([lambdarange[0], lambdarange[-1]])

        ax[1,0].set_xlim([logmassrange[0], logmassrange[-1]])
        
        ax[1,0].ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True)


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
        contours = ax[1,0].contour(normed_posterior, t_contours, extent=[logmassrange[0],logmassrange[-1], lambdarange[0],lambdarange[-1]], colors='white', linewidths=0.5)
        
        ########################################################################################################################
        ########################################################################################################################
        


        lambda_logposterior = special.logsumexp(log_posterior, axis=1)

        normalisedlambdaposterior = np.exp(lambda_logposterior-special.logsumexp(lambda_logposterior))

        cdflambdaposterior = np.cumsum(normalisedlambdaposterior)
        meanlabda = lambdarange[np.abs(norm.cdf(0)-cdflambdaposterior).argmin()]
        lambdapercentiles = []
        for zscore in zscores:
            lambdapercentiles.append(lambdarange[np.abs(norm.cdf(zscore)-cdflambdaposterior).argmin()])

        print(f'\n\n\n{np.min(np.abs(meanlabda-lambdapercentiles))/np.sqrt(1e8/(eventmultiplier*totalevents))*1e5}*1e-5 \n\n')
        ax[1,1].plot(lambdarange,normalisedlambdaposterior, c='tab:green')

        ax[1,1].axvline(meanlabda, c='tab:green', ls=':')


        for o, percentile in enumerate(lambdapercentiles):
                    color = colormap(np.abs(zscores[o])/4-0.01)

                    ax[1,1].axvline(percentile, c=color, ls=':')
        ax[1,1].axvline(truelambda, ls='--', color="tab:orange")
        ax[1,1].set_xlabel(r'$\lambda$')
        ax[1,1].set_ylim([0, None])
        ax[1,1].set_xlim([lambdarange[0], lambdarange[-1]])

        
        ax[1,1].ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True)
        
        # ax11_2 = ax[1,1].secondary_xaxis('top')
        # ax11_2.set_xlabel(r'$\langle \sigma v \rangle$ [unit]')
        # ax11_2.set_xticks(ax[1,1].get_xticks(), labels=0.5*np.asarray(ax[1,1].get_xticks()))
        # import matplotlib.ticker as ticker
        
        # ax11_2.xaxis.set_major_formatter(ticker.ScalarFormatter.ticklabel_format(style='sci', useMathText=True))

        plt.savefig(time.strftime(f"{stemdirectory}/{totalevents}events_lm{truelogmass}_l{truelambda}_%m%d_%H%M.pdf"))
        plt.show()
        
        if 0:
            
            # Checking contours
            onesigma_contour_vertices = contours.allsegs[2][0]
            onesigma_logmassvals = onesigma_contour_vertices[:,0]
            onesigma_lambdavals = onesigma_contour_vertices[:,1]
            
            twosigma_contour_vertices = contours.allsegs[1][0]
            twosigma_logmassvals = twosigma_contour_vertices[:,0]
            twosigma_lambdavals = twosigma_contour_vertices[:,1]
            
            threesigma_contour_vertices = contours.allsegs[0][0]
            threesigma_logmassvals = threesigma_contour_vertices[:,0]
            threesigma_lambdavals = threesigma_contour_vertices[:,1]
            
            
            
            # sigmavvals = convert_lambda_to_sigmav(lambdavals, totalevents*eventmultiplier, logmassvals, log10eaxistrue, longitudeaxistrue, latitudeaxistrue, aeff, 52.5*60*60, 1, fulllogDMmodel=None, fulllogDMspectra=None, DMdiffJfactor_func=None)
            
            
            
            plt.figure()
            plt.pcolormesh(logmassrange, lambdarange, np.exp(log_posterior))
            plt.plot(onesigma_logmassvals, onesigma_lambdavals, c='white')
            plt.plot(twosigma_logmassvals, twosigma_lambdavals, c='red')
            plt.plot(threesigma_logmassvals, threesigma_lambdavals, c='yellow')
            plt.show()
            
            sigmavvals = convert_lambda_to_sigmav(twosigma_lambdavals, totalevents*eventmultiplier, twosigma_logmassvals, log10eaxistrue, longitudeaxistrue, latitudeaxistrue, aefffunc, 52.5*60*60, 1, fulllogDMmodel=None, fulllogDMspectra=energymassinputspectralfunc)

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
            plt.axhline(2e-26, ls='--', c='grey', alpha=0.5, label=r'DarkSUSY thermal $\langle \sigma v \rangle$')
            plt.ylim([1e-27, 1e-24])
            plt.xlim([1e-1,1e2])
            plt.grid(color='grey', alpha=0.2)
            plt.loglog()
            plt.xlabel(r'$m_\chi$ [TeV]')
            plt.ylabel(r'$\langle \sigma v \rangle$ [cm$^3$ s$^{-1}$]')
            plt.text(1.1e-1, 1.1e-27, 'signal: Einasto, Scalar Singlet \nbackground: CR + IEM', fontsize=6)
            plt.legend(fontsize=6)
            plt.tight_layout()
            plt.savefig('Figures/sensitivity_plot_31_08_23.png')
            plt.show()
    if showsamples:
        
        signal_log10e_measured,  signal_lon_measured, signal_lat_measured = np.load(f"{stemdirectory}/1/meassigsamples.npy")
        bkg_log10e_measured, bkg_lon_measured, bkg_lat_measured = np.load(f"{stemdirectory}/1/measbkgsamples.npy")
        siglogevals, siglonvals, siglatvals = np.load(f"{stemdirectory}/1/truesigsamples.npy")
        bkglogevals, bkglonvals, bkglatvals= np.load(f"{stemdirectory}/1/truebkgsamples.npy")
        
        
        for rundir in rundirs[1:]:
            runnum = rundir.replace(stemdirectory+'/', '')
            print("runnum: ", runnum)
            params              = np.load(f"{rundir}/params.npy")
            siglogevals_temp, siglonvals_temp, siglatvals_temp = np.load(f"{rundir}/truesigsamples.npy")
            bkglogevals_temp, bkglonvals_temp, bkglatvals_temp= np.load(f"{rundir}/truebkgsamples.npy")
            
            signal_log10e_measured_temp,  signal_lon_measured_temp, signal_lat_measured_temp = np.load(f"{rundir}/meassigsamples.npy")
            bkg_log10e_measured_temp, bkg_lon_measured_temp, bkg_lat_measured_temp = np.load(f"{rundir}/measbkgsamples.npy")

            siglogevals = np.concatenate(siglogevals, siglogevals_temp) 
            siglonvals = np.concatenate(siglonvals, siglonvals_temp) 
            siglatvals = np.concatenate(siglatvals, siglatvals_temp) 
            
            bkglogevals = np.concatenate(bkglogevals, bkglogevals_temp) 
            bkglonvals = np.concatenate(bkglonvals, bkglonvals_temp) 
            bkglatvals = np.concatenate(bkglatvals, bkglatvals_temp) 
            
            signal_log10e_measured = np.concatenate(signal_log10e_measured, signal_log10e_measured_temp) 
            signal_lon_measured = np.concatenate(signal_lon_measured, signal_lon_measured_temp) 
            signal_lat_measured = np.concatenate(signal_lat_measured, signal_lat_measured_temp) 
            
            bkg_log10e_measured = np.concatenate(bkg_log10e_measured, bkg_log10e_measured_temp) 
            bkg_lon_measured = np.concatenate(bkg_lon_measured, bkg_lon_measured_temp) 
            bkg_lat_measured = np.concatenate(bkg_lat_measured, bkg_lat_measured_temp) 
            
        
        
        
        measured_log10e = list(signal_log10e_measured)+list(bkg_log10e_measured)
        measured_lon = list(signal_lon_measured)+list(bkg_lon_measured)
        measured_lat = list(signal_lat_measured)+list(bkg_lat_measured)
        
        from matplotlib.colors import LogNorm
        
        plt.figure(figsize=(5,4))
        plt.hist2d(measured_lon, measured_lat, bins=[longitudeaxis+2e-10, latitudeaxis+2e-10], edgecolor='face')

        plt.colorbar()
        plt.xlabel('Longitude [deg]')
        plt.ylabel('Latitude [deg]')
        plt.savefig('Figures/realistic_spatial_samples.pdf')
        plt.show()
        
        
        plt.figure(figsize=(5,4))
        plt.hist(10**np.array(measured_log10e), bins=10**log10eaxistrue-1e-10)
        plt.xscale('log')
        plt.yscale('log')

        plt.xlim(np.min(10**np.array(measured_log10e)))
        plt.xlabel('Energy [TeV]')
        plt.savefig('Figures/realistic_energy_samples.pdf')
        plt.show()
        
        