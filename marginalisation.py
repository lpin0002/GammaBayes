
from utils import inverse_transform_sampling, axis, bkgdist, makedist, edisp, eaxis_mod
from scipy import integrate, special, interpolate, stats
import numpy as np
import os, time, random
from tqdm import tqdm
import matplotlib.pyplot as plt

sigsamples          = np.load("truesigsamples.npy")
sigsamples_measured = np.load("meassigsamples.npy")
bkgsamples          = np.load("truebkgsamples.npy")
bkgsamples_measured = np.load("measbkgsamples.npy")
truevals            = np.concatenate((sigsamples, bkgsamples))
measuredvals        = np.concatenate((sigsamples_measured,bkgsamples_measured))



logmassrange = np.linspace(1.09,1.11,21)
lambdarange = np.linspace(0.89,0.91,21)

edispnorms = np.array([special.logsumexp(edisp(axis,axisval)+eaxis_mod) for axisval in axis])

edisplist = []
for sample in measuredvals:
        edisplist.append(edisp(sample,axis)-edispnorms)
edisplist = np.array(edisplist)

logmassposterior = []
bkgdistnormed = bkgdist(axis) - special.logsumexp(bkgdist(axis)+eaxis_mod)
for lambdaval in tqdm(lambdarange):
    templambdarow = []
    for logmass in logmassrange:
            tempsigdist = makedist(logmass)
            tempsigdistaxis = tempsigdist(axis) - special.logsumexp(tempsigdist(axis)+eaxis_mod)

            tempmargval = 0
            for i, sample in enumerate(measuredvals):
                    tempval = np.logaddexp(np.log(lambdaval)+tempsigdistaxis,np.log(1-lambdaval)+bkgdistnormed)
                    # print(tempval)
                    tempmargval += special.logsumexp(tempval+edisplist[i]+eaxis_mod)
            
            templambdarow.append(tempmargval)
    logmassposterior.append(templambdarow)


normedlogposterior = logmassposterior - special.logsumexp(logmassposterior)
np.save("normedlogposterior.npy", normedlogposterior)
plt.figure()
plt.pcolormesh(logmassrange, lambdarange, np.exp(normedlogposterior))
plt.axvline(1.1, c='r')
plt.axhline(0.9, c='r')
if measuredvals.shape[0]>20000:
       plt.title(str(measuredvals.shape[0]))
       plt.savefig(time.strftime(f"posterior%H%M_{measuredvals.shape[0]}.pdf"))
plt.savefig("posterior.pdf")
plt.show()

plt.figure()
plt.plot(logmassrange, np.exp(normedlogposterior[np.abs(0.5-lambdarange).argmin(),:]))
plt.axvline(1.1, c='r')
plt.savefig("logmassslice.pdf")
if measuredvals.shape[0]>20000:
       plt.title(str(measuredvals.shape[0]))
       plt.savefig(time.strftime(f"logmassslice%H%M_{measuredvals.shape[0]}.pdf"))
plt.show()


plt.figure()
plt.plot(lambdarange, np.exp(normedlogposterior[:, np.abs(0.5-logmassrange).argmin()]))
plt.xlabel('lambda')
plt.axvline(0.9,c='r')
if measuredvals.shape[0]>20000:
       plt.title(str(measuredvals.shape[0]))
       plt.savefig(time.strftime(f"lambdaslice%H%M_{measuredvals.shape[0]}.pdf"))
plt.savefig("lambdaslice.pdf")
plt.show()
