import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, interpolate, integrate, special
from tqdm import tqdm
import time
import os
import sys
from utils import make_gaussian, backgrounddist, energydisp, axis, edispnorms
from scipy.sparse import dok_matrix

# The identifier for the run you are analyzing
timestring = sys.argv[1]

pseudomeasuredenergysamples_background  = np.load(f"runs/{timestring}/measured_background.npy")
pseudomeasuredenergysamples_signal      = np.load(f"runs/{timestring}/measured_signal.npy")
axis                                    = np.load(f"runs/{timestring}/axis.npy")

measuredsamples = np.concatenate((pseudomeasuredenergysamples_signal, pseudomeasuredenergysamples_background))

del(pseudomeasuredenergysamples_background)
del(pseudomeasuredenergysamples_signal)


logmassrange = np.linspace(0.5,1.5, 300).astype(np.float128)



edisplist = []
for sample in measuredsamples:
    edisplist.append(energydisp(sample, axis)/edispnorms)
edisplist = np.array(edisplist).astype(np.float128)

eaxis = np.power(10., axis).astype(np.float128)


###############################################################################################################################################
# Marginalising with signal distribution
###############################################################################################################################################
marglist_signal = []
for logmass in tqdm(logmassrange, desc="signal marginalisation"):

    marglist_singlemass = []
    singlemass_sigfunc = make_gaussian(centre=logmass, axis=axis)
    singlemass_sigfunc_norm = np.sum(singlemass_sigfunc(axis)).astype(np.float128)
    

    for i, sample in enumerate(measuredsamples):
        singlemassfunceval = singlemass_sigfunc(sample).astype(np.float128)
        edispmultiple=np.multiply(edisplist[i],eaxis).astype(np.float128)
        singledatumposterior = singlemassfunceval*edispmultiple/singlemass_sigfunc_norm

        marglist_singlemass.append(np.sum(singledatumposterior))
    
    marglist_signal.append(marglist_singlemass)

marglist_signal = np.array(marglist_signal).astype(np.float128)



###############################################################################################################################################
# Marginalising with background distribution
###############################################################################################################################################
marglist_background =  []
backgroundnorm = np.sum(backgrounddist(axis)).astype(np.float128)
for i, sample in tqdm(enumerate(measuredsamples), desc="background marginalisation"):
    edispmultiple=np.multiply(edisplist[i]/backgroundnorm,eaxis).astype(np.float128)
    marglist_background.append(np.sum(backgrounddist(sample)*edispmultiple).astype(np.float128))

marglist_background = np.array(marglist_background).astype(np.float128)

print("Background: ", marglist_background.shape)
print("Signal: ", marglist_signal.shape)


###############################################################################################################################################
# Creating the mixture with lambda
###############################################################################################################################################
lambdalist = np.linspace(0.,1., 300)


print(marglist_signal.shape)
fullmarglist_signal = []
fullmarglist_background = []
marglist_background_repeated = np.repeat(marglist_background,logmassrange.shape[0]).reshape((marglist_background.shape[0], logmassrange.shape[0])).T

del(measuredsamples)
del(edispnorms)
del(edisplist)
del(eaxis)

print("marglist shape: ", marglist_signal.shape)
print("repeated background shape: ", marglist_background_repeated.shape)

print("original background: ", marglist_background[:5])
print("new background list: ", marglist_background_repeated[0,:5])


###############################################################################################################################################
# Doing the product of all the data points
###############################################################################################################################################
for lval in tqdm(lambdalist, desc="iterating through lambdas"):
    fullmarglist_signal.append(np.log(lval*marglist_signal).astype(np.float128))
    fullmarglist_background.append(np.log((1-lval)*marglist_background_repeated).astype(np.float128))



###############################################################################################################################################
# Calculating the unnnormalised posterior
###############################################################################################################################################

unnormalisedposterior = np.ones((lambdalist.shape[0], logmassrange.shape[0]), dtype=np.float128)
print("Calculating the unnormalised posterior...")
for i, lambdaval in tqdm(enumerate(lambdalist), desc="lambda iteration"):
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
normalisedposterior = np.exp(unnormalisedposterior-posteriornormalisationfactor)
print("Finished normalising.\033[F")
print(normalisedposterior.max())





###############################################################################################################################################
# Saving the results
###############################################################################################################################################
print("Now saving results. \n1...\033[F")
np.save(f"runs/{timestring}/posteriorarray.npy", normalisedposterior)
print("2...\033[F")
np.save(f"runs/{timestring}/logmassrange.npy", logmassrange)
print("3...\033[F")
np.save(f"runs/{timestring}/lambdarange.npy", lambdalist)
print("Done! Hope the posteriors look good my g.")
print(timestring)