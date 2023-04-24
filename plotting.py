from utils import inverse_transform_sampling, axis, bkgdist, makedist, edisp, eaxis_mod
from scipy import integrate, special, interpolate, stats
import os, time, random, sys, numpy as np, matplotlib.pyplot as plt, chime, warnings
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

# sigsamples          = np.load(f"data/{identifier}/truesigsamples.npy")
# sigsamples_measured = np.load(f"data/{identifier}/meassigsamples.npy")
# bkgsamples          = np.load(f"data/{identifier}/truebkgsamples.npy")
# bkgsamples_measured = np.load(f"data/{identifier}/measbkgsamples.npy")

currentdirecyory = os.getcwd()
stemdirectory = currentdirecyory+f'/data/{identifier}'
print("stem directory: ", stemdirectory)

rundirs = [x[0] for x in os.walk(stemdirectory)][1:]
print("list of run directories: ", rundirs)
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
# truevals            = np.concatenate((sigsamples, bkgsamples))
# measuredvals        = np.concatenate((sigsamples_measured,bkgsamples_measured))



print(lambdarange)
normedlogposterior = np.load(f"data/{identifier}/normedlogposterior{integrationtype}.npy")


plt.figure(dpi=100)
pcol = plt.pcolor(lambdarange, logmassrange, np.exp(normedlogposterior), snap=True)
pcol.set_edgecolor('face')
plt.ylabel("log mass [TeV]")
plt.xlabel("lambda = signal events/total events")
plt.colorbar(label="Probability Density [1/TeV]")
plt.axhline(truelogmass, c='r')
plt.axvline(truelambdaval, c='r')
plt.grid(False)
if totalevents>10000:
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
if totalevents>10000:
       plt.title(str(totalevents))
       plt.savefig(time.strftime(f"data/{identifier}/logmassslice%H%M_{totalevents}{integrationtype}.pdf"))
plt.show()


plt.figure()
plt.plot(lambdarange, np.exp(normedlogposterior[np.abs(truelogmass-logmassrange).argmin(),:]))
plt.xlabel("lambda = signal events/total events")
plt.ylabel("Probability density (slice) []")
plt.axvline(truelambdaval,c='r', label=params[1,0])
plt.legend()
if totalevents>10000:
       plt.title(str(totalevents))
       plt.savefig(time.strftime(f"data/{identifier}/lambdaslice%H%M_{totalevents}{integrationtype}.pdf"))
plt.savefig(f"Figures/LatestFigures/lambdaslice{integrationtype}.pdf")
plt.show()