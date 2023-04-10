
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import interpolate
from tqdm import tqdm
import time
import os
import sys
from utils import make_gaussian, backgrounddist, energydisp, inv_trans_sample, axis, find_closest,edispnorms


lambdaval           = np.float128(sys.argv[1])
Nsamples            = np.float128(sys.argv[2])
signalcentreval     = np.float128(sys.argv[3])
timestring          = str(sys.argv[4])

Nsamples_signal     = int(np.round(lambdaval*Nsamples))
Nsamples_background = int(np.round((1-lambdaval)*Nsamples))


signaldist = make_gaussian(centre = signalcentreval,axis=axis)

print("Sampling the signal distribution...")
truesamples_signal = axis[inv_trans_sample(Nsamples_signal, np.multiply(signaldist(axis), np.power(10.,axis)).astype(np.float128))]
print("Sampling the background distribution...")
truesamples_background = axis[inv_trans_sample(Nsamples_background, np.multiply(backgrounddist(axis), np.power(10.,axis)).astype(np.float128))]
print("Finished sampling the true distributions.")


print("Creating pseudo-measured signal values...")
pseudomeasuredenergysamples_signal = []
for sample in truesamples_signal:
    pseudomeasuredenergysamples_signal.append(np.float128(axis[inv_trans_sample(1,np.multiply(energydisp(sample, axis)/edispnorms,np.power(10.,axis)).astype(np.float128))]))
pseudomeasuredenergysamples_signal = np.array(pseudomeasuredenergysamples_signal).astype(np.float128)

print("Creating psuedo-measured background values...")
pseudomeasuredenergysamples_background = []
for sample in truesamples_background:
    pseudomeasuredenergysamples_background.append(np.float128(axis[inv_trans_sample(1,np.multiply(energydisp(sample, axis)/edispnorms,np.power(10.,axis)))]))
pseudomeasuredenergysamples_background = np.array(pseudomeasuredenergysamples_background).astype(np.float128)
print("Finished.")


print(timestring)

try:
    os.mkdir('runs')
except:
    print("Runs folder already exists")

try:
    os.mkdir(f'runs/{timestring}')
except:
    print("Beware, the folder for this run already seems to exist.")

print(pseudomeasuredenergysamples_background.shape)
print(pseudomeasuredenergysamples_signal.shape)

np.save(f"runs/{timestring}/measured_background.npy", pseudomeasuredenergysamples_background)
np.save(f"runs/{timestring}/measured_signal.npy", pseudomeasuredenergysamples_signal)
np.save(f"runs/{timestring}/true_background.npy", truesamples_background)
np.save(f"runs/{timestring}/true_signal.npy", truesamples_signal)
np.save(f"runs/{timestring}/axis.npy", axis)

np.save(f"runs/{timestring}/parameterarray", np.array([['logmass [TeV]', 'true lambda', 'Nsamples'],[signalcentreval, lambdaval, Nsamples]]))