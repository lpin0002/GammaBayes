from utils import log10eaxis, bkgdist, makedist, edisp, logjacob
import numpy as np
import os, time, sys
#import chime
from BFCalc.BFInterp import DM_spectrum_setup
from brutesampler import brutedynesty


if __name__ == '__main__':
        signal_setup_function = DM_spectrum_setup

        try:
                identifier = sys.argv[1]
        except:
                identifier = time.strftime("%d%m%H")
        try:
                numcores = int(sys.argv[2])
        except:
                numcores = 10
                

        sigdistsetup = DM_spectrum_setup
        # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
        np.seterr(divide='ignore', invalid='ignore')

        currentdirecyory = os.getcwd()
        stemdirectory = currentdirecyory+f'/data/{identifier}'
        print(stemdirectory)

        rundirs = [x[0] for x in os.walk(stemdirectory)][1:]

        print(rundirs)
        print(len(rundirs))
        runnum=1
        print("runnum: ", runnum)
        params              = np.load(f"data/{identifier}/{runnum}/params.npy", allow_pickle=True)
        sigsamples_measured = list(np.load(f"data/{identifier}/{runnum}/meassigsamples.npy"))
        bkgsamples_measured = list(np.load(f"data/{identifier}/{runnum}/measbkgsamples.npy"))

        dontincludenums = []
        for runnum in range(2,len(rundirs)+1):
                if not(runnum in dontincludenums):
                        print("runnum: ", runnum)
                        paramstmp             = np.load(f"data/{identifier}/{runnum}/params.npy", allow_pickle=True)
                        print(paramstmp[1,:].shape)
                        assert not np.sum(paramstmp[1,:].astype(float)-params[1,:].astype(float))

                        sigsamples_measured  +=list(np.load(f"data/{identifier}/{runnum}/meassigsamples.npy"))
                        bkgsamples_measured  +=list(np.load(f"data/{identifier}/{runnum}/measbkgsamples.npy"))

        measured_events = np.array(sigsamples_measured+bkgsamples_measured)
        print("Yes it's running up to here")
        # brutedynesty(measuredevents, logsigpriorsetup, logbkgprior, logedisp, log10eaxis=log10eaxis, nlive = 1000, print_progress=False):
        results = brutedynesty(measured_events, sigdistsetup, bkgdist, edisp, numcores=numcores)

        np.save(f'data/{identifier}/results_brute.npy', results)