import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, interpolate, integrate, special
from tqdm import tqdm
import time
import os
import sys
from utils import make_gaussian, backgrounddist, energydisp, axis, edispnorms
from scipy.sparse import dok_matrix
import chime

# The identifier for the run you are analyzing
timestring = sys.argv[1]

pseudomeasuredenergysamples_background  = np.load(f"runs/{timestring}/measured_background.npy")
pseudomeasuredenergysamples_signal      = np.load(f"runs/{timestring}/measured_signal.npy")
axis                                    = np.load(f"runs/{timestring}/axis.npy")

measuredsamples = np.concatenate((pseudomeasuredenergysamples_signal, pseudomeasuredenergysamples_background))

del(pseudomeasuredenergysamples_background)
del(pseudomeasuredenergysamples_signal)


logmassrange = np.linspace(axis[0],axis[-1],axis.shape[0]-1)


edisplist = []
for sample in measuredsamples:
    edisplist.append(energydisp(sample, axis)-np.log(edispnorms))
edisplist = np.array(edisplist).astype(np.float128)


eaxis = np.power(10., axis).astype(np.float128)
eaxis_mod = np.log(eaxis)
print("edispnorms: ", edispnorms)
print("edisplist: ", edisplist)

###############################################################################################################################################
# Marginalising with signal distribution
###############################################################################################################################################
marglist_signal = []
for logmass in tqdm(logmassrange, desc="signal marginalisation"):

    marglist_singlemass = []
    singlemass_sigfunc = make_gaussian(centre=logmass, axis=axis)
    singlemass_sigfunc_norm = special.logsumexp(singlemass_sigfunc(axis))
    

    for i, sample in enumerate(measuredsamples):
        singlemassfunceval = singlemass_sigfunc(sample)
        singledatumposterior = singlemassfunceval+edisplist[i]-singlemass_sigfunc_norm+eaxis_mod

        marglist_singlemass.append(special.logsumexp(singledatumposterior))
    
    marglist_signal.append(marglist_singlemass)

marglist_signal = np.array(marglist_signal)
print(marglist_signal)


###############################################################################################################################################
# Marginalising with background distribution
###############################################################################################################################################
marglist_background =  []
backgroundnorm = special.logsumexp(backgrounddist(axis).astype(np.float128))
for i, sample in tqdm(enumerate(measuredsamples), desc="background marginalisation"):
    marglist_background.append(special.logsumexp(backgrounddist(sample) + edisplist[i] - backgroundnorm + eaxis_mod))

marglist_background = np.array(marglist_background)
# print(marglist_background)

print("Background: ", marglist_background.shape)
print("Signal: ", marglist_signal.shape)


###############################################################################################################################################
# Creating the mixture with lambda
###############################################################################################################################################
lambdalist = np.linspace(0.85,0.95,11)


print(marglist_signal.shape)
fullmarglist_signal = []
fullmarglist_background = []
marglist_background_repeated = np.repeat(marglist_background,logmassrange.shape[0]).reshape((marglist_background.shape[0], logmassrange.shape[0])).T


print("marglist shape: ", marglist_signal.shape)
print("repeated background shape: ", marglist_background_repeated.shape)

print("original background: ", marglist_background[:5])
print("new background list: ", marglist_background_repeated[0,:5])


###############################################################################################################################################
# Doing the product of all the data points
###############################################################################################################################################
print("Multiplying models by lambda and (1-lambda)...")
for lval in tqdm(lambdalist, desc="iterating through lambdas"):
    fullmarglist_signal.append(np.log(lval)+marglist_signal)
    fullmarglist_background.append(np.log(1-lval)+marglist_background_repeated)



###############################################################################################################################################
# Calculating the unnnormalised posterior
###############################################################################################################################################

unnormalisedposterior = np.ones((lambdalist.shape[0], logmassrange.shape[0]), dtype=np.float128)
print("Calculating the unnormalised posterior...")
for i, lambdaval in tqdm(enumerate(lambdalist), desc="lambda progress"):
    for j, logmass in enumerate(logmassrange):
        addition = np.sum(np.logaddexp(fullmarglist_signal[i][j,:],fullmarglist_background[i][j,:]),dtype=np.float128)
        # print(addition)
        unnormalisedposterior[i,j] = addition.astype(np.float128)
print("Finished calculating the unnormalised posterior.")


print("Finding max before normalisation...")
print(unnormalisedposterior.max())
print("Finished finding max.\033[F")



###############################################################################################################################################
# Calculating the normalisation of the posterior (and thus posterior)
###############################################################################################################################################
print("Calculating normalisation factor...")
posteriornormalisationfactor = special.logsumexp(unnormalisedposterior)
print("Finished calculation of normalisation. Now normalising...")
normalisedposterior = unnormalisedposterior-posteriornormalisationfactor
print("Finished normalising.\033[F")
print(normalisedposterior.max())





###############################################################################################################################################
# Saving the results
###############################################################################################################################################
print("Now saving results. \n1...")
np.save(f"runs/{timestring}/posteriorarray.npy", normalisedposterior)
print("\033[F2...")
np.save(f"runs/{timestring}/logmassrange.npy", logmassrange)
print("\033[F3...")
np.save(f"runs/{timestring}/lambdarange.npy", lambdalist)
chime.success()
chime.info('sonic')
print("\033[F Done! Hope the posteriors look good my g.")
print(timestring)