import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, interpolate, integrate, special
from tqdm import tqdm
import time
import os
import sys
from utils import make_gaussian, backgrounddist, energydisp, axis, edispnorms
from scipy.sparse import dok_matrix
# import chime

# The identifier for the run you are analyzing
timestring = sys.argv[1]

# Using the identifier to access the measured samples and the log energy axis we used for the run(s).
pseudomeasuredenergysamples_background  = np.load(f"runs/{timestring}/measured_background.npy")
pseudomeasuredenergysamples_signal      = np.load(f"runs/{timestring}/measured_signal.npy")
axis                                    = np.load(f"runs/{timestring}/axis.npy")

# Concatenating the measured samples as we always work with them together anyway.
measuredsamples = np.concatenate((pseudomeasuredenergysamples_signal, pseudomeasuredenergysamples_background))


# This code has had some problems with memory usage in the past, and we don't won't to keep anything on ram that
    # we don't need to. As these aren't used again we remove them for these reasons.
del(pseudomeasuredenergysamples_background)
del(pseudomeasuredenergysamples_signal)

# The way the subdivisions in the code below works is that say you have a range of values 0,1,2,3,4,5,6,7,8,9,10
    # If you then wanted to increase the number of divisions, effectively putting in an extra sample in the middle
    # of each pair of the previous values you'd have 0,0.5,1.0,1.5,2.0,2.5,...,8.5,9.0,9.5,10.0. So the number of
    # samples hasn't actually doubled, it's increased by 2*(previous number -1)+1. That is what I replicate in 
    # general below.
logenergyaxis_subdivisions= 3
logmassrange = np.linspace(axis[0],axis[-1],logenergyaxis_subdivisions*(axis.shape[0]-1)+1)


# Below I make a pseudo-lookup table of the value of the energy dispersion for each measured sample so that during the 
    # marginalisation steps it isn't evaluated multiple times unnecessarily. The edispnorms are the log of normalisation
    # factors for each value of _true energy_ where we want the total probability for the whole _measured energy_ axis
    # to be one as that is how the energy dispersion is normalised.

edisplist = []
for sample in measuredsamples:
    edisplist.append(energydisp(sample, axis)-edispnorms)
edisplist = np.array(edisplist).astype(np.float128)

# The axis we are effectively sampling is the log energy axis, hence the ''axis'' is in log energy. But our integration
    # must occur over _energy_ so we need to map the axis values to the energy values with np.power(10.,_value_). We then 
    # take the _natural_ log of this with np.log as we are primarily working in log probability space to avoid numerical
    # instability problems.
eaxis = np.power(10., axis).astype(np.float128)
eaxis_mod = np.log(eaxis)
print("edispnorms: ", edispnorms)
print("edisplist: ", edisplist)

###############################################################################################################################################
# Marginalising with signal distribution
###############################################################################################################################################
# We instantiate a python list for the marginalisation results for the signal
marglist_signal = []

# For this marginalisation, we must marginalise the events for all the different possible values of mass/logmass, hence this iteration
for logmass in tqdm(logmassrange, desc="signal marginalisation", ncols=80):

    # We instantiate a dummy list for storing the marginalisation results for a single signal model/logmass/mass
    marglist_singlemass = []

    #Instantiating the signal model for that particular mass.
    singlemass_sigfunc = make_gaussian(centre=logmass, axis=axis)

    # Calculating the normalisation for the signal function
    singlemass_sigfunc_norm = special.logsumexp(singlemass_sigfunc(axis))
    

    for i, sample in enumerate(measuredsamples):

        # Evaluating the probability of the sample given the signal model
        singlemassfunceval = singlemass_sigfunc(sample)

        # Evaluating the probability of the measured value given the prior and likelihood for the 
            # range of true energies
        singledatumposterior = singlemassfunceval+edisplist[i]-singlemass_sigfunc_norm+eaxis_mod

        # Using logsumexp for the actual marginalisation
        marglist_singlemass.append(special.logsumexp(singledatumposterior))
    
    marglist_signal.append(marglist_singlemass)

marglist_signal = np.array(marglist_signal)
print(marglist_signal)


###############################################################################################################################################
# Marginalising with background distribution
###############################################################################################################################################
marglist_background =  []
backgroundnorm = special.logsumexp(backgrounddist(axis).astype(np.float128))
for i, sample in tqdm(enumerate(measuredsamples), desc="background marginalisation", ncols=80):
    marglist_background.append(special.logsumexp(backgrounddist(sample) + edisplist[i] - backgroundnorm + eaxis_mod))

marglist_background = np.array(marglist_background)
# print(marglist_background)

print("Background: ", marglist_background.shape)
print("Signal: ", marglist_signal.shape)


###############################################################################################################################################
# Creating the mixture with lambda
###############################################################################################################################################
lambdalist = np.linspace(0.,1.0,51)


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
for lval in tqdm(lambdalist, desc="iterating through lambdas", ncols=80):
    fullmarglist_signal.append(np.log(lval)+marglist_signal)
    fullmarglist_background.append(np.log(1-lval)+marglist_background_repeated)



###############################################################################################################################################
# Calculating the unnnormalised posterior
###############################################################################################################################################

unnormalisedposterior = np.ones((lambdalist.shape[0], logmassrange.shape[0]), dtype=np.float128)
print("Calculating the unnormalised posterior...")
for i, lambdaval in tqdm(enumerate(lambdalist), desc="lambda progress"):
    for j, logmass in enumerate(logmassrange):
        product = np.sum(np.logaddexp(fullmarglist_signal[i][j,:],fullmarglist_background[i][j,:]),dtype=np.float128)
        # print(addition)
        unnormalisedposterior[i,j] = product
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
# chime.success()
# chime.info('sonic')
print("\033[F Done! Hope the posteriors look good my g.")
print(timestring)