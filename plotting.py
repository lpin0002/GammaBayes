from utils import inverse_transform_sampling, bkgdist, makedist, edisp, eaxis_mod, log10eaxis
from scipy import integrate, special, interpolate, stats
import os, time, random, sys, numpy as np, matplotlib.pyplot as plt, warnings, corner.corner as corner
from matplotlib import cm
import matplotlib as mpl
from scipy.stats import norm
from tqdm import tqdm
from BFCalc.BFInterp import DM_spectrum_setup


try:
    identifier = sys.argv[1]
except:
    identifier = time.strftime("%d%m%H")
try:
    showhyperparameterposterior = int(sys.argv[2])
except:
    showhyperparameterposterior = 0
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
    
print("Integration type: ", integrationtype)
integrationtype = "_"+integrationtype.lower()



whattoplot = [showhyperparameterposterior,shownuisanceparameterposterior,showsamples]

currentdirecyory = os.getcwd()
stemdirectory = currentdirecyory+f'/data/{identifier}'
print("\nstem directory: ", stemdirectory, '\n')

rundirs = [x[0] for x in os.walk(stemdirectory)][1:]
print("number of run directories: ", len(rundirs), '\n')

if 'brute' in integrationtype:
        params               = np.load(f"{rundirs[0]}/params.npy")
        
        totalevents          = int(params[1,1])
        truelambda           = float(params[1,0])
        truelogmass          = float(params[1,2])


        print(f"{params[0,0]} = {params[1,0]}")
        print(f"{params[0,2]} = {params[1,2]}")
        
        
        print(f"Total events: {totalevents}\n")
        
        if showhyperparameterposterior:
        
                # recyclingresults     = np.load(f'{stemdirectory}/recyclingresults.npy', allow_pickle=True)
                
                brutesamplerresults = np.load(f'data/{identifier}/results_brute.npy', allow_pickle=True)

                recyclingresults = brutesamplerresults.item()
                runsamples = recyclingresults.samples_equal()


                figure = corner(
                            runsamples,
                            quantiles=[0.025, 0.16, 0.84, 0.975],
                            levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                            labels=[r"log$_{10}$ $m_\chi$", r"$\lambda$"],
                            # show_titles=True,
                            title_kwargs={"fontsize": 12},
                            bins = 32,
                            truths=[truelogmass, truelambda],
                            labelpad=-0.1,
                            tick_kwargs={'rotation':90},
                            color='#0072C1',
                            truth_color='tab:orange',
                            plot_density=0, 
                            plot_datapoints=True, 
                            fill_contours=True,
                            max_n_ticks=7,
                            hist_kwargs=dict(density=True),
                            smooth=0.9,
                            # smooth1d=0.9
                )
                plt.suptitle(f"Nevents = {totalevents}", size=16)
                figure.set_size_inches(8,8)
                figure.set_dpi(100)
                #plt.tight_layout()
                
                plt.savefig(time.strftime(f'{stemdirectory}/Hyperparameter_Posterior_%H_direct.pdf'))
                plt.show()


if integrationtype=='_direct':
    params               = np.load(f"{rundirs[0]}/params.npy")
    totalevents          = int(params[1,1])
    truelambdaval           = float(params[1,0])
    truelogmass          = float(params[1,2])
    truesigsamples    = np.load(f"{rundirs[0]}/truesigsamples.npy")
    # truebkgsamples    = np.load(f"{rundirs[0]}/truebkgsamples.npy")
    meassigsamples    = np.load(f"{rundirs[0]}/meassigsamples.npy")
    # measbkgsamples    = np.load(f"{rundirs[0]}/measbkgsamples.npy")
    logmassrange = np.load(f'{rundirs[0]}/logmassrange{integrationtype}.npy')
    # lambdarange = np.load(f'{rundirs[0]}/lambdarange{integrationtype}.npy')

    truesamples             = np.array(list(truesigsamples)+list([]))
    meassamples          = np.array(list(meassigsamples)+list([]))
    for rundir in rundirs[1:]:
            runnum = rundir.replace(stemdirectory+'/', '')
            print("runnum: ", runnum)
            params              = np.load(f"data/{identifier}/{runnum}/params.npy")
            logmassrange = np.load(f'data/{identifier}/{runnum}/logmassrange{integrationtype}.npy')
            # lambdarange = np.load(f'data/{identifier}/{runnum}/lambdarange{integrationtype}.npy')
            edisplist = np.load(f'data/{identifier}/{runnum}/edisplist{integrationtype}.npy')
            # bkgmarglist = np.load(f'data/{identifier}/{runnum}/bkgmarglist{integrationtype}.npy')
            # sigmarglogzvals = np.load(f'data/{identifier}/{runnum}/sigmarglogzvals{integrationtype}.npy')
            params              = np.load(f"data/{identifier}/{runnum}/params.npy")
            truesigsamples       =np.concatenate((truesigsamples, np.load(f"{rundir}/truesigsamples.npy")))
            # truebkgsamples       =np.concatenate((truebkgsamples, np.load(f"{rundir}/truebkgsamples.npy")))
            # meassigsamples       =np.concatenate((meassigsamples, np.load(f"{rundir}/meassigsamples.npy")))
            # measbkgsamples       =np.concatenate((measbkgsamples, np.load(f"{rundir}/measbkgsamples.npy")))
            params[1,:]         = params[1,:]
            truelogmass     = float(params[1,2])
            nevents         = int(params[1,1])
            totalevents+=nevents
            truelambdaval   = float(params[1,0])


    logmass_logposterior = np.load(f"data/{identifier}/logmass_logposterior{integrationtype}.npy")
    
        

    # print(special.logsumexp(normedlogposterior))
    # plt.figure(dpi=100)
    # # logmassrange, lambdarange, 
    # pcol = plt.pcolor(logmassrange, lambdarange, np.exp(normedlogposterior).T, snap=True)
    # pcol.set_edgecolor('face')

    # # Plot the contours
    
    # mean = np.mean(np.exp(normedlogposterior).T)
    # std = np.std(np.exp(normedlogposterior).T)
    # contour_levels = [mean + std, mean + 2*std, mean + 3*std]
    # levels =[1. - np.exp(-0.5), 1. - np.exp(-2), 1. - np.exp(-9 / 2.)],
    # plt.contour(logmassrange, lambdarange, np.exp(normedlogposterior).T, contour_levels, cmap='autumn')

    # # plt.legend()
    
    
    # plt.xlabel(r"$log_{10}$(mass) [TeV]")
    # plt.ylabel("lambda = signal events/total events")
    # plt.colorbar(pcol, label="Probability Density [1/TeV]")
    # plt.axvline(truelogmass, c='tab:pink')
    # plt.axhline(truelambdaval, c='tab:pink')
    # plt.grid(False)
    # plt.title(f"{totalevents} total events")
    # plt.savefig(time.strftime(f"data/{identifier}/posterior%H_{totalevents}{integrationtype}.pdf"))
    # plt.savefig(f"Figures/LatestFigures/posterior{integrationtype}.pdf")
    # plt.show()
    
    

    colormap = cm.get_cmap('Blues_r', 4)


    deltalogmass = (logmassrange[1]-logmassrange[0])
    normedposterior = np.exp(logmass_logposterior-special.logsumexp(logmass_logposterior+np.log(deltalogmass)))
    cdfposterior = np.cumsum(normedposterior*deltalogmass)
    print(cdfposterior[-1])
    mean = logmassrange[np.abs(0.5-cdfposterior).argmin()]
    zscores = [-3, -2,-1,1,2, 3]
    percentiles = []
    for zscore in zscores:
        percentiles.append(logmassrange[np.abs(norm.cdf(zscore)-cdfposterior).argmin()])



    plt.figure(dpi=160, figsize=(10,5))
    plt.title(f'Nevents = {totalevents}')
    plt.plot(logmassrange, normedposterior, c='tab:green')
    plt.axvline(mean, c='tab:green', ls='--', alpha=1, label='mean')
    for o, percentile in enumerate(percentiles):
        color = colormap(np.abs(zscores[o])/4-0.01)

        plt.axvline(percentile, c=color, ls=':')
        

    plt.axvline(truelogmass, c='tab:orange', ls='-.', label='true logmass value', alpha=0.8)
    plt.xlabel(r'log$_{10}$ mass [TeV]')
    norm = mpl.colors.Normalize(vmin=0, vmax=5)

    cb1 = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ticks=np.arange(1,5))
    cb1.set_label(r'standard deviations')
    plt.legend()
    plt.savefig(time.strftime(f'data/{identifier}/logmassposterior_%m%d_%H%M_logmass={truelogmass}.png'))
    plt.show()


    # plt.figure()
    # plt.plot(lambdarange, np.sum(np.exp(normedlogposterior),axis=0))
    # plt.xlabel("lambda = signal events/total events")
    # plt.ylabel("Probability density (slice) []")
    # plt.axvline(truelambdaval,c='r', label=params[1,0])
    # plt.legend()
    # plt.title(str(totalevents))
    # plt.ylim([0, None])
    # plt.xlim([lambdarange[0], lambdarange[-1]])
    # plt.savefig(time.strftime(f"data/{identifier}/lambdaslice%H_{totalevents}{integrationtype}.pdf"))
    # plt.savefig(f"Figures/LatestFigures/lambdaslice{integrationtype}.pdf")
    # plt.show()


if whattoplot[2]:

    centrevals = log10eaxis[:-1:6]+0.001*(log10eaxis[1]-log10eaxis[0])
    

    plt.figure()
    plt.title("true values")
    truesightvals = plt.hist(truesigsamples, bins=centrevals, alpha=0.7, label='True sig samples', color='forestgreen')
    truebkghtvals = plt.hist(truebkgsamples, bins=centrevals, alpha=0.7, label='True bkg samples', color='royalblue')
    plt.axvline(truelogmass, label=r'true $log_{10}(m_\chi)$ [TeV]', c="tab:orange")
    plt.xlabel(r'True $log_{10}(E)$ [TeV]')
    plt.ylabel('Number of samples')
    plt.legend()
    plt.savefig("Figures/LatestFigures/TrueVals.pdf")
    plt.show()




    plt.figure()
    plt.title("measured values")
    meassightvals = plt.hist(meassigsamples, bins=centrevals, alpha=0.7, label='pseudo-measured sig samples', color='forestgreen')
    measbkghtvals = plt.hist(measbkgsamples, bins=centrevals, alpha=0.7, label='pseudo-measured bkg samples', color='royalblue')
    plt.axvline(truelogmass, label=r'true $log_{10}(m_\chi)$ [TeV]', c="tab:orange")
    plt.xlabel(r'Reconstructed (Measured) $log_{10}(E)$ [TeV]')
    plt.ylabel('Number of samples')
    plt.legend()
    plt.savefig("Figures/LatestFigures/MeasuredVals.pdf")
    plt.show()
    
