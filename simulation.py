
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import interpolate
from tqdm import tqdm
import time
import os
import sys
from utils import make_gaussian, backgrounddist, energydisp, inv_trans_sample, axis, find_closest


lambdaval = float(sys.argv[1])
Nsamples = float(sys.argv[2])
signalcentreval = float(sys.argv[3])
timestring = str(sys.argv[4])

Nsamples_signal = int(np.round(lambdaval*Nsamples))
Nsamples_background = int(np.round((1-lambdaval)*Nsamples))


signaldist = make_gaussian(centre = signalcentreval,axis=axis)

# plt.figure(dpi=200)
# plt.plot(axis,signaldist(axis), label="Signal")
# plt.plot(axis, backgrounddist(axis), label="Background")
# plt.legend()
# plt.show()




# edispplot = []
# for energy_measured in axis:
#     edisprow = []
#     for energy_true in axis:
#         edisprow.append(energydisp(energy_measured, energy_true))
#     edispplot.append(edisprow)


# plt.figure(dpi=100)
# plt.pcolormesh(axis,axis, edispplot)
# plt.colorbar()
# plt.clim([np.min(edispplot),np.max(edispplot)])
# plt.show()




truesamples_signal = axis[inv_trans_sample(Nsamples_signal, np.multiply(signaldist(axis), np.power(10.,axis)))]
truesamples_background = axis[inv_trans_sample(Nsamples_background, np.multiply(backgrounddist(axis), np.power(10.,axis)))]


# plt.figure(dpi=100)
# histogramdata = plt.hist(truesamples_signal, bins=axis, label="Signal")
# histogramdata2 = plt.hist(truesamples_background, bins=axis, label="Background")
# plt.plot(axis,signaldist(axis)/max(signaldist(axis))*max(histogramdata[0]))
# plt.legend()
# plt.show()



pseudomeasuredenergysamples_signal = []
for sample in truesamples_signal:
    pseudomeasuredenergysamples_signal.append(float(axis[inv_trans_sample(1,np.multiply(energydisp(sample, axis),np.power(10.,axis)))]))
pseudomeasuredenergysamples_signal = np.array(pseudomeasuredenergysamples_signal)

pseudomeasuredenergysamples_background = []
for sample in truesamples_background:
    pseudomeasuredenergysamples_background.append(float(axis[inv_trans_sample(1,np.multiply(energydisp(sample, axis),np.power(10.,axis)))]))
pseudomeasuredenergysamples_background = np.array(pseudomeasuredenergysamples_background)


# plt.figure(dpi=100)
# histogramdata = plt.hist(pseudomeasuredenergysamples_signal, bins=axis)
# histogramdata2 = plt.hist(pseudomeasuredenergysamples_background, bins=axis)
# plt.plot(axis,signaldist(axis)/max(signaldist(axis))*max(histogramdata[0]))
# plt.show()

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