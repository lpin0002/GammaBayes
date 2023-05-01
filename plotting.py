from utils import inverse_transform_sampling, axis, bkgdist, makedist, edisp, eaxis_mod
from scipy import integrate, special, interpolate, stats
import os, time, random, sys, numpy as np, matplotlib.pyplot as plt, chime, warnings, corner.corner as corner
from tqdm import tqdm
from BFCalc.BFInterp import DM_spectrum_setup


try:
    identifier = sys.argv[1]
except:
    identifier = time.strftime("%d%m%H")
try:
       integrationtype = sys.argv[2]
except:
       integrationtype = "nested"

integrationtype = "_"+integrationtype.lower()


currentdirecyory = os.getcwd()
stemdirectory = currentdirecyory+f'/data/{identifier}'
print("\nstem directory: ", stemdirectory, '\n')

rundirs = [x[0] for x in os.walk(stemdirectory)][1:]
print("number of run directories: ", len(rundirs), '\n')

if integrationtype=='_nested':
       params               = np.load(f"{rundirs[0]}/params.npy")
       # bkgmargresults       = np.load(f'{rundirs[0]}/bkgmargresults.npy', allow_pickle=True)
       # propmargresults      = np.load(f'{rundirs[0]}/propmargresults.npy', allow_pickle=True)
       totalevents          = int(params[1,1])
       truelambda           = float(params[1,0])
       truelogmass          = float(params[1,2])
       truesigsamples    = np.load(f"{rundirs[0]}/truesigsamples.npy")
       truebkgsamples    = np.load(f"{rundirs[0]}/truebkgsamples.npy")
       meassigsamples    = np.load(f"{rundirs[0]}/meassigsamples.npy")
       measbkgsamples    = np.load(f"{rundirs[0]}/measbkgsamples.npy")

       truesamples             = np.array(list(truesigsamples)+list(truebkgsamples))
       meassamples          = np.array(list(meassigsamples)+list(measbkgsamples))

       for rundir in rundirs[1:]:
              # bkgmargresults       = np.concatenate((bkgmargresults,  np.load(f'{rundir}/bkgmargresults.npy', allow_pickle=True)))
              # propmargresults      = np.concatenate((propmargresults,np.load(f'{rundir}/propmargresults.npy', allow_pickle=True)))
              tempparams           = np.load(f"{rundir}/params.npy")
              totalevents          += int(tempparams[1,1])
              truesigsamples       =np.concatenate((truesigsamples, np.load(f"{rundir}/truesigsamples.npy")))
              truebkgsamples       =np.concatenate((truebkgsamples, np.load(f"{rundir}/truebkgsamples.npy")))
              meassigsamples       =np.concatenate((meassigsamples, np.load(f"{rundir}/meassigsamples.npy")))
              measbkgsamples       =np.concatenate((measbkgsamples, np.load(f"{rundir}/measbkgsamples.npy")))
              truetempsamples    = np.concatenate((np.load(f"{rundir}/truesigsamples.npy"),np.load(f"{rundir}/truebkgsamples.npy")))
              meastempsamples    = np.concatenate((np.load(f"{rundir}/meassigsamples.npy"),np.load(f"{rundir}/measbkgsamples.npy")))

              truesamples          = np.concatenate((truesamples, truetempsamples))
              meassamples          = np.concatenate((meassamples, meastempsamples))

       
       print(f"Total events: {totalevents}\n")
       
       recyclingresults     = np.load(f'{stemdirectory}/recyclingresults.npy', allow_pickle=True)

       recyclingresults = recyclingresults.item()
       runsamples = recyclingresults.samples_equal()


       figure = corner(
              runsamples,
              labels=[r"log$_{10}$ $m_\chi$", r"$\lambda$"],
              show_titles=True,
              title_kwargs={"fontsize": 12},
              bins = [25,25],
              truths=[truelogmass, truelambda],
              labelpad=-0.1,
              tick_kwargs={'rotation':90},
              color='#0072C1',
              truth_color='tab:orange',
              plot_density=False, 
              plot_datapoints=True, 
              fill_contours=True,
              max_n_ticks=3, 
              #hist_kwargs=dict(density=True),
              smooth=0.9,
              #smooth1d=0.5
       )
       # plt.suptitle(f"Nevents = {totalevents}", size=16)
       figure.set_size_inches(8,8)
       figure.set_dpi(200)
       #plt.tight_layout()
       
       plt.savefig(time.strftime(f'{stemdirectory}/Hyperparameter_Posterior_%H.pdf'))
       plt.show()


       # sampleindex = -3
       # bkgsamples = bkgmargresults[sampleindex].samples_equal()

       # print(runsamples)

       # figure = corner(
       #        bkgsamples,
       #        # labels=[r"log$_{10}$ $m_\chi$"],
       #        show_titles=True,
       #        title_kwargs={"fontsize": 12},
       #        bins = [250],
       #        truths=[truesamples[sampleindex]],
       #        labelpad=-0.2,
       #        truth_color='r',
       #        range=[(axis[np.abs(axis-truesamples[sampleindex]).argmin()-16],axis[np.abs(axis-truesamples[sampleindex]).argmin()+16])]
       # )
       # # plt.suptitle(f"Nevents = {totalevents}", size=16)
       # figure.set_size_inches(5,5)
       # figure.set_dpi(200)
       # #plt.tight_layout()
       
       # plt.savefig(time.strftime(f'{stemdirectory}/TrueEnergy_Posterior_%H.pdf'))
       # plt.show()




if integrationtype=='direct':

       totalevents = 0
       for rundir in rundirs:
              runnum = rundir.replace(stemdirectory+'/', '')
              print("runnum: ", runnum)
              params              = np.load(f"data/{identifier}/{runnum}/params.npy")
              logmassrange = np.load(f'data/{identifier}/{runnum}/logmassrange{integrationtype}.npy')
              lambdarange = np.load(f'data/{identifier}/{runnum}/lambdarange{integrationtype}.npy')
              edisplist = np.load(f'data/{identifier}/{runnum}/edisplist{integrationtype}.npy')
              bkgmarglist = np.load(f'data/{identifier}/{runnum}/bkgmarglist{integrationtype}.npy')
              sigmarglogzvals = np.load(f'data/{identifier}/{runnum}/sigmarglogzvals{integrationtype}.npy')
              logmassrange = np.load(f'data/{identifier}/{runnum}/logmassrange{integrationtype}.npy')
              lambdarange = np.load(f'data/{identifier}/{runnum}/lambdarange{integrationtype}.npy')
              params              = np.load(f"data/{identifier}/{runnum}/params.npy")
              params[1,:]         = params[1,:]
              truelogmass     = float(params[1,2])
              nevents         = int(params[1,1])
              totalevents+=nevents
              truelambdaval   = float(params[1,0])


       print(lambdarange)
       normedlogposterior = np.load(f"data/{identifier}/normedlogposterior{integrationtype}.npy")


       plt.figure(dpi=100)
       pcol = plt.pcolor(lambdarange, logmassrange, np.exp(normedlogposterior), snap=True)
       pcol.set_edgecolor('face')
       plt.ylabel(r"$log_{10}$(mass) [TeV]")
       plt.xlabel("lambda = signal events/total events")
       plt.colorbar(label="Probability Density [1/TeV]")
       plt.axhline(truelogmass, c='r')
       plt.axvline(truelambdaval, c='r')
       plt.grid(False)
       plt.title(str(totalevents))
       plt.savefig(time.strftime(f"data/{identifier}/posterior%H%M_{totalevents}{integrationtype}.pdf"))
       plt.savefig(f"Figures/LatestFigures/posterior{integrationtype}.pdf")
       plt.show()

       plt.figure()
       plt.plot(logmassrange, np.exp(normedlogposterior[:,np.abs(truelambdaval-lambdarange).argmin()]))
       plt.axvline(truelogmass, c='r', label=params[1,2])
       plt.xlabel("log mass [TeV]")
       plt.ylabel("Probability density (slice) [1/TeV]")
       plt.legend()
       plt.savefig(f"Figures/LatestFigures/logmassslice{integrationtype}.pdf")
       plt.title(str(totalevents))
       plt.savefig(time.strftime(f"data/{identifier}/logmassslice%H%M_{totalevents}{integrationtype}.pdf"))
       plt.show()


       plt.figure()
       plt.plot(lambdarange, np.exp(normedlogposterior[np.abs(truelogmass-logmassrange).argmin(),:]))
       plt.xlabel("lambda = signal events/total events")
       plt.ylabel("Probability density (slice) []")
       plt.axvline(truelambdaval,c='r', label=params[1,0])
       plt.legend()
       plt.title(str(totalevents))
       plt.savefig(time.strftime(f"data/{identifier}/lambdaslice%H%M_{totalevents}{integrationtype}.pdf"))
       plt.savefig(f"Figures/LatestFigures/lambdaslice{integrationtype}.pdf")
       plt.show()



centrevals = axis[:-1]+0.5*(axis[1]-axis[0])
# backgroundintegrals = []
# signalintegrals = []
# for i in range(len(axis[1:])):
#     evals = np.linspace(10**axis[i],10**axis[i+1],100)
#     signalintegrals.append(np.exp(special.logsumexp(sigdist(np.log10(evals))+np.log(evals))))
#     backgroundintegrals.append(np.exp(special.logsumexp(bkgdist(np.log10(evals))+np.log(evals))))
# signalintegrals = np.array(signalintegrals)
# signalintegrals = np.array(signalintegrals)

plt.figure()
plt.title("true values")
truesightvals = plt.hist(truesigsamples, bins=axis[::5]-0.01, alpha=0.7, label='True sig samples', color='forestgreen')
truebkghtvals = plt.hist(truebkgsamples, bins=axis[::5]-0.01, alpha=0.7, label='True bkg samples', color='royalblue')
plt.axvline(truelogmass, label=r'true $log_{10}(m_\chi)$ [TeV]', c="tab:orange")
plt.xlabel(r'True $log_{10}(E)$ [TeV]')
plt.ylabel('Number of samples')
plt.legend()
plt.savefig("Figures/LatestFigures/TrueVals.pdf")
plt.show()



# plt.figure()
# plt.title("background true values")
# # tmeashist = plt.hist(bkgsamples, bins=centrevals, alpha=0.7, label="Measured background")
# # bkgdistvals = np.exp(bkgdist(axis))*eaxis
# # plt.plot(axis, bkgdistvals/np.max(bkgdistvals)*np.max(bkghistvals[0]), label='point background with jacobian')
# # plt.plot(centrevals, backgroundintegrals/np.max(backgroundintegrals)*np.max(bkghistvals[0]), label='background integral vals')

# plt.legend()
# plt.savefig("Figures/LatestFigures/TrueValsBackground.pdf")
# plt.show()


plt.figure()
plt.title("measured values")
meassightvals = plt.hist(meassigsamples, bins=axis[::5]-0.01, alpha=0.7, label='pseudo-measured sig samples', color='forestgreen')
measbkghtvals = plt.hist(measbkgsamples, bins=axis[::5]-0.01, alpha=0.7, label='pseudo-measured bkg samples', color='royalblue')
plt.axvline(truelogmass, label=r'true $log_{10}(m_\chi)$ [TeV]', c="tab:orange")
plt.xlabel(r'Reconstructed (Measured) $log_{10}(E)$ [TeV]')
plt.ylabel('Number of samples')
plt.legend()
plt.savefig("Figures/LatestFigures/MeasuredVals.pdf")
plt.show()