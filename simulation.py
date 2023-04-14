
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import interpolate
from tqdm import tqdm
import time
import os
import sys
from utils import make_gaussian, backgrounddist, energydisp, inv_trans_sample, axis, find_closest,edispnorms
# import chime

lambdaval           = np.float128(sys.argv[1])
Nsamples            = np.float128(sys.argv[2])
signalcentreval     = np.float128(sys.argv[3])
timestring          = str(sys.argv[4])

Nsamples_signal     = int(np.round(lambdaval*Nsamples))
Nsamples_background = int(np.round((1-lambdaval)*Nsamples))


signaldist = make_gaussian(centre = signalcentreval,axis=axis)

eaxis = np.power(10.,axis)
eaxis_mod = np.log(eaxis)

print("Sampling the signal distribution...")
truesamples_signal = axis[inv_trans_sample(Nsamples_signal, np.exp(signaldist(axis)+eaxis_mod).astype(np.float128))]
print("\033[F Sampling the background distribution...")
truesamples_background = axis[inv_trans_sample(Nsamples_background, np.exp(backgrounddist(axis)+eaxis_mod).astype(np.float128))]
print("\033[F Finished sampling the true distributions.")

# TODO: Inverses transform sampling with log probabilities
print("Creating pseudo-measured signal values...")
pseudomeasuredenergysamples_signal = []
for sample in truesamples_signal:
    pseudomeasuredenergysamples_signal.append(np.float128(axis[inv_trans_sample(1,np.exp(energydisp(sample, axis) - edispnorms+eaxis_mod).astype(np.float128))]))
pseudomeasuredenergysamples_signal = np.array(pseudomeasuredenergysamples_signal).astype(np.float128)

print("\033[F Creating psuedo-measured background values...")
pseudomeasuredenergysamples_background = []
for sample in truesamples_background:
    pseudomeasuredenergysamples_background.append(np.float128(axis[inv_trans_sample(1,np.exp(energydisp(sample, axis) - edispnorms+eaxis_mod).astype(np.float128))]))
pseudomeasuredenergysamples_background = np.array(pseudomeasuredenergysamples_background).astype(np.float128)
print("\033[F Finished creating pseudo-measured values.")


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

# chime.success()
# chime.info('sonic')
print("Finished script, please input into marginalisation.py for analysis.")