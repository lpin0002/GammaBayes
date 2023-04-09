import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, interpolate, integrate, special
from tqdm import tqdm
import time
import os
import sys
from utils import make_gaussian, backgrounddist, energydisp, inv_trans_sample, axis

# The identifier for the run you are analyzing
timestring = sys.argv[1]

pseudomeasuredenergysamples_background  = np.load(f"runs/{timestring}/measured_background.npy")
pseudomeasuredenergysamples_signal      = np.load(f"runs/{timestring}/measured_signal.npy")
truesamples_background                  = np.load(f"runs/{timestring}/true_background.npy")
truesamples_signal                      = np.load(f"runs/{timestring}/true_signal.npy")
axis                                    = np.load(f"runs/{timestring}/axis.npy")

measuredsamples = np.concatenate((pseudomeasuredenergysamples_signal, pseudomeasuredenergysamples_background))


logmassrange = np.linspace(axis[0],axis[-1],400)
# Doing the marginalisations with the signal distribution
# p(E_m|S)


edisplist = []
for sample in measuredsamples:
    edisplist.append(energydisp(sample, axis))


eaxis = np.power(10., axis)

marglist_signal = []
for logmass in tqdm(logmassrange, desc="signal marginalisation"):

    marglist_singlemass = []
    singlemass_sigfunc = make_gaussian(centre=logmass, axis=axis)
    singlemass_sigfunc_norm = sum(singlemass_sigfunc(axis))

    for i, sample in enumerate(measuredsamples):

        marglist_singlemass.append(integrate.simps(y=singlemass_sigfunc(sample)*np.multiply(edisplist[i],eaxis)/singlemass_sigfunc_norm, 
                                                                                                            x=axis))
    
    marglist_signal.append(marglist_singlemass)

marglist_signal = np.array(marglist_signal)

print(marglist_signal)




# Doing the marginalisation with the background distribution
# p(E_m|B)
marglist_background =  []

for sample in tqdm(measuredsamples, desc="background marginalisation"):
    marglist_background.append(integrate.simps(y=backgrounddist(sample)*np.multiply(energydisp(sample, axis),np.power(10.,axis)), x=axis))
marglist_background = np.array(marglist_background)

print("Background: ", marglist_background.shape)
print("Signal: ", marglist_signal.shape)

# We then do the mixture
# p(\lambda|E_m, S, B) = prod( \lambda*p(E_m|S) + (1-\lambda)*p(E_m|B))*prior(\lambda)
lambdalist = np.linspace(0,1,int(axis.shape[0]))


print(marglist_signal.shape)
fullmarglist_signal = []
fullmarglist_background = []
marglist_background_repeated = np.repeat(marglist_background,logmassrange.shape[0]).reshape((marglist_background.shape[0], logmassrange.shape[0])).T

print("marglist shape: ", marglist_signal.shape)
print("repeated background shape: ", marglist_background_repeated.shape)

print("original background: ", marglist_background[:5])
print("new background list: ", marglist_background_repeated[0,:5])

for lval in tqdm(lambdalist, desc="iterating through lambdas"):
    fullmarglist_signal.append(lval*marglist_signal)
    fullmarglist_background.append((1-lval)*marglist_background_repeated)

# fullmarglist_signal = np.outer(marglist_signal,lambdalist)
# fullmarglist_background = np.outer(marglist_background, (1-lambdalist))
fullmarglist_background = np.array(fullmarglist_background)
fullmarglist_signal = np.array(fullmarglist_signal)

print(fullmarglist_background.shape)
print(fullmarglist_signal.shape)

unnormalisedposterior = np.nansum(np.log(fullmarglist_signal+fullmarglist_background), axis=2)
print(unnormalisedposterior.max())


normalisedposterior = np.exp(unnormalisedposterior-special.logsumexp(unnormalisedposterior))
print(normalisedposterior.max())

np.save(f"runs/{timestring}/posteriorarray.npy", normalisedposterior)
np.save(f"runs/{timestring}/logmassrange.npy", logmassrange)
np.save(f"runs/{timestring}/lambdarange.npy", lambdalist)