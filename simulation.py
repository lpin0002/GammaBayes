from scipy import integrate, special, interpolate, stats
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from utils import inverse_transform_sampling, axis, makedist, edisp, bkgdist


sigdist = makedist(1.0)

nevents = 10000
sigsamples = axis[inverse_transform_sampling(sigdist(axis),nevents)]

sigsamples_measured = []
for sigsample in tqdm(sigsamples, desc="Creating measured signal vals..."):
    sigsamples_measured.append(axis[inverse_transform_sampling(edisp(axis,sigsample),Nsamples=1)])
sigsamples_measured = np.array(sigsamples_measured)


bkgsamples = axis[inverse_transform_sampling(bkgdist(axis),nevents)]

bkgsamples_measured = []
for bkgsample in tqdm(bkgsamples, desc="Creating measured signal vals..."):
    bkgsamples_measured.append(axis[inverse_transform_sampling(edisp(axis,bkgsample),Nsamples=1)])
bkgsamples_measured = np.array(bkgsamples_measured)



plt.figure()
plt.title("true values")
centrevals = axis[:-1]+0.5*(axis[1]-axis[0])

sighistvals = plt.hist(sigsamples, bins=centrevals, alpha=0.7)
sigdistvals = np.exp(sigdist(axis))
plt.plot(axis, sigdistvals/np.max(sigdistvals)*np.max(sighistvals[0]))

bkghistvals = plt.hist(bkgsamples, bins=centrevals, alpha=0.7)
bkgdistvals = np.exp(bkgdist(axis))
plt.plot(axis, bkgdistvals/np.max(bkgdistvals)*np.max(bkghistvals[0]))
plt.show()


plt.figure()
plt.title("measured values")
plt.hist(sigsamples_measured, bins=centrevals, alpha=0.7)
plt.hist(bkgsamples_measured, bins=centrevals, alpha=0.7)
plt.show()


np.save("truesigsamples.npy", sigsamples)
np.save("meassigsamples.npy", sigsamples_measured)
np.save("truebkgsamples.npy", bkgsamples)
np.save("measbkgsamples.npy", bkgsamples_measured)