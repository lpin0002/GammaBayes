from utils3d import *
from BFCalc.createspectragrids import darkmatterdoubleinput
from scipy import special
import os, time, sys, numpy as np, matplotlib.pyplot as plt, corner.corner as corner
from matplotlib import cm
import matplotlib as mpl
from scipy.stats import norm
from tqdm import tqdm


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
try:
    integrationtype = str(sys.argv[6])
except:
    integrationtype = "direct"
    
    
specsetup = darkmatterdoubleinput
    
print("Integration type: ", integrationtype)
integrationtype = "_"+integrationtype.lower()



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
    truesigsamples          = np.load(f"{rundirs[0]}/truesigsamples.npy")
    truebkgsamples          = np.load(f"{rundirs[0]}/truebkgsamples.npy", allow_pickle= True)
    meassigsamples          = np.load(f"{rundirs[0]}/meassigsamples.npy")
    measbkgsamples          = np.load(f"{rundirs[0]}/measbkgsamples.npy", allow_pickle=True)
    
    if showhyperparameterposterior:
        logmassrange = np.load(f'{stemdirectory}/logmassrange{integrationtype}.npy')
        lambdarange = np.load(f'{stemdirectory}/lambdarange{integrationtype}.npy')

    # lambdarange = np.load(f'{rundirs[0]}/lambdarange{integrationtype}.npy')

    truesamples             = np.array(list(truesigsamples)+list(meassigsamples))
    meassamples          = np.array(list(meassigsamples)+list(measbkgsamples))
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
        eventmultiplier = 1.0
        
        
        
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
        ax[1,0].pcolormesh(logmassrange, lambdarange, np.exp(log_posterior), shading='gouraud')
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
        ax[1,0].contour(normed_posterior, t_contours, extent=[logmassrange[0],logmassrange[-1], lambdarange[0],lambdarange[-1]], colors='white', linewidths=0.5)
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


        plt.savefig(time.strftime(f"{stemdirectory}/{totalevents}events_lm{truelogmass}_l{truelambda}_%m%d_%H%M.pdf"))
        plt.show()
        
    if showsamples:
        
        signal_log10e_measured,  signal_lon_measured, signal_lat_measured = np.load(f"{stemdirectory}/1/meassigsamples.npy")
        bkg_log10e_measured, bkg_lon_measured, bkg_lat_measured = np.load(f"{stemdirectory}/1/measbkgsamples.npy")
        siglogevals, siglonvals, siglatvals = np.load(f"{stemdirectory}/1/truesigsamples.npy")
        bkglogevals, bkglonvals, bkglatvals= np.load(f"{stemdirectory}/1/truebkgsamples.npy")
        
        
        plt.figure()
        plt.hist2d(bkglonvals, bkglatvals, bins=[plotlongitudeaxistrue, plotlatitudeaxistrue])
        plt.show()
        
        