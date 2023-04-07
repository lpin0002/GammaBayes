import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, interpolate, integrate, special
from tqdm import tqdm
import time
import os
import sys
from utils import make_gaussian, backgrounddist, signaldist, energydisp, inv_trans_sample, axis

# The identifier for the run you are analyzing
timestring = sys.argv[1]

pseudomeasuredenergysamples_background  = np.load(f"runs/{timestring}/measured_background.npy")
pseudomeasuredenergysamples_signal      = np.load(f"runs/{timestring}/measured_signal.npy")
truesamples_background                  = np.load(f"runs/{timestring}/true_background.npy")
truesamples_signal                      = np.load(f"runs/{timestring}/true_signal.npy")
axis                                    = np.load(f"runs/{timestring}/axis.npy")

measuredsamples = np.concatenate((pseudomeasuredenergysamples_signal, pseudomeasuredenergysamples_background))

# Doing the marginalisations with the signal distribution
# p(E_m|S)
marglist_signal = []
for sample in measuredsamples:
    marglist_signal.append(integrate.simps(y=signaldist(sample)*np.multiply(energydisp(sample, axis),np.power(10.,axis)), x=axis))
marglist_signal = np.array(marglist_signal)

print(marglist_signal)

continueq = input("Do you wish to continue?: ")
if 'N' in continueq.upper():
    raise Exception("You did not wish to continue.")
# Doing the marginalisation with the background distribution
# p(E_m|B)
marglist_background =  []

for sample in tqdm(measuredsamples):
    marglist_background.append(integrate.simps(y=backgrounddist(sample)*np.multiply(energydisp(sample, axis),np.power(10.,axis)), x=axis))
marglist_background = np.array(marglist_background)

print(marglist_background)


continueq = input("Do you wish to continue?: ")
if 'N' in continueq.upper():
    raise Exception("You did not wish to continue.")
# We then do the mixture
# p(\lambda|E_m, S, B) = prod( \lambda*p(E_m|S) + (1-\lambda)*p(E_m|B))*prior(\lambda)
lambdalist = np.linspace(0,1,2000)


print(marglist_signal.shape)
fullmarglist_signal = []
fullmarglist_background = []

fullmarglist_signal = np.outer(marglist_signal,lambdalist)
fullmarglist_background = np.outer(marglist_background, (1-lambdalist))


unnormalisedposterior = np.nansum(np.log(fullmarglist_signal+fullmarglist_background), axis=0)
print(unnormalisedposterior.max())


normalisedposterior = np.exp(unnormalisedposterior-special.logsumexp(unnormalisedposterior))
print(normalisedposterior.max())


plt.figure()
plt.plot(lambdalist, normalisedposterior, c="r", marker="s", linestyle="None", markersize=0.1)
plt.axvline(0.5)
plt.xlabel(r"$\lambda$")
plt.savefig(f"{timestring}_posterior_lambda_slice.png")
plt.show()