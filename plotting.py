import matplotlib.pyplot as plt
import numpy as np
import os, sys
from tqdm import tqdm

# The identifier for the run you are analyzing
timestring      = sys.argv[1]
params          = np.load(f"runs/{timestring}/parameterarray.npy")
signalcentreval = float(params[1,0])
truelambda      = float(params[1,1])


pseudomeasuredenergysamples_background  = np.load(f"runs/{timestring}/measured_background.npy")
pseudomeasuredenergysamples_signal      = np.load(f"runs/{timestring}/measured_signal.npy")
truesamples_background                  = np.load(f"runs/{timestring}/true_background.npy")
truesamples_signal                      = np.load(f"runs/{timestring}/true_signal.npy")
axis                                    = np.load(f"runs/{timestring}/axis.npy")

print(params)
print("Nsamples: ", pseudomeasuredenergysamples_background.shape[0]+pseudomeasuredenergysamples_signal.shape[0])



plt.figure()
plt.hist(pseudomeasuredenergysamples_background, bins=axis)
plt.hist(pseudomeasuredenergysamples_signal, bins=axis)
plt.xlabel("Log E [TeV]")
plt.savefig(f"runs/{timestring}/pseudomeasuredsamples.png")
plt.savefig(f"runs/{timestring}/pseudomeasuredsamples.pdf")
plt.show()

plt.figure()
plt.hist(np.power(10.,truesamples_background), bins=np.power(10.,axis))
plt.hist(np.power(10.,truesamples_signal), bins=np.power(10.,axis))
plt.xlabel("E [TeV]")
plt.savefig(f"runs/{timestring}/truesamples.png")
plt.savefig(f"runs/{timestring}/truesamples.pdf")
plt.show()


normalisedposterior                     = np.load(f"runs/{timestring}/posteriorarray.npy")
logmassrange                            = np.load(f"runs/{timestring}/logmassrange.npy")
lambdarange                             = np.load(f"runs/{timestring}/lambdarange.npy")


plt.figure()
plt.pcolormesh(logmassrange.astype(float), lambdarange.astype(float), normalisedposterior, cmap='inferno')
plt.axvline(signalcentreval, lw=0.5, c='g')
plt.axhline(truelambda, lw=0.5, c='g')
plt.savefig(f"runs/{timestring}/posterior.png")
plt.savefig(f"runs/{timestring}/posterior.pdf")
plt.colorbar()
plt.show()

plt.figure()
plt.plot(lambdarange, normalisedposterior[:, np.abs(logmassrange - signalcentreval).argmin()], lw=0.5)
plt.axvline(truelambda, lw=0.5)
plt.savefig(f"runs/{timestring}/lambdaslice.png")
plt.savefig(f"runs/{timestring}/lambdaslice.pdf")
plt.show()


plt.figure()
plt.plot(logmassrange, normalisedposterior[np.abs(lambdarange - truelambda).argmin(),:], lw=0.5)
plt.axvline(signalcentreval, lw=0.5, c='g', label="actual val")
plt.axvline(axis[(np.abs(axis-signalcentreval)).argmin()], label="Closest val on axis")
plt.axvline(axis[(np.abs(axis-signalcentreval)).argmin()]+(axis[1]-axis[0]), label="Closest val on axis to the right")
plt.axvline(axis[(np.abs(axis-signalcentreval)).argmin()]-(axis[1]-axis[0]), label="Closest val on axis to the left")

plt.legend()
plt.savefig(f"runs/{timestring}/logmassslice.png")
plt.savefig(f"runs/{timestring}/logmassslice.pdf")
plt.show()