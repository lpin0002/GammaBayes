from utils import log10eaxis, bkgdist, makedist, edisp, logjacob
import numpy as np
import os, time, sys
#import chime
from BFCalc.BFInterp import DM_spectrum_setup
from brutesampler import brutedynesty


if __name__ == '__main__':
        specsetup = DM_spectrum_setup

        try:
                identifier = sys.argv[1]
        except:
                identifier = time.strftime("%d%m%H")
        try:
                numcores = int(sys.argv[2])
        except:
                numcores = 10
                

        # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
        np.seterr(divide='ignore', invalid='ignore')

        currentdirecyory = os.getcwd()
        stemdirectory = currentdirecyory+f'/data/{identifier}'
        print(stemdirectory)

        rundirs = [x[0] for x in os.walk(stemdirectory)][1:]
        firstrundirectory = rundirs[0]


        print(rundirs)
        print(len(rundirs))
        
        runnum=1
        print("runnum: ", runnum)
         # Loading in values
        logirfvals = np.load(f"{stemdirectory}/irfvalues.npy")
        truelambda, Nsamples, truelogmassval = np.load(f"{firstrundirectory}/params.npy")[1,:]
        truelogmassval = float(truelogmassval)
        truelambda = float(truelambda)
        totalevents = int(Nsamples)*len(rundirs)


        print("Yes it's running up to here")
        # brutedynesty(measuredevents, logsigpriorsetup, logbkgprior, logedisp, log10eaxis=log10eaxis, nlive = 1000, print_progress=False):
        results =brutedynesty(specsetup, bkgdist, logirfvals, numcores=numcores)

        np.save(f'data/{identifier}/results_brute.npy', results)