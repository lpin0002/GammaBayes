
import os, sys, numpy as np, time
from scipy import special, stats
from tqdm import tqdm
from runrecycle import runrecycle
from utils import logpropdist, makedist, log10eaxis
from BFCalc.BFInterp import DM_spectrum_setup
# import chime
if __name__ == '__main__':
    # chime.info('sonic')
    try:
        identifier = sys.argv[1]
    except:
        identifier = time.strftime("%d%m%H")
    try:
        reccores = int(sys.argv[2])
    except:
        reccores = int(10)

    sigdistsetup = makedist

    


    currentdirecyory = os.getcwd()
    stemdirectory = currentdirecyory+f'/data/{identifier}'
    print(f'Stem directory: {stemdirectory}')

    rundirs = [x[0] for x in os.walk(stemdirectory)][1:]

    print(rundirs)
    print(f'Number of runs: {len(rundirs)}')
    runaccess_startingindex=1
    print("runnum: ", rundirs[0])
    try:
        params              = np.load(f"{rundirs[0]}/params.npy")
        bkgmargresults = np.load(f'{rundirs[0]}/bkgmargresults.npy', allow_pickle=True)
        propmargresults = np.load(f'{rundirs[0]}/propmargresults.npy', allow_pickle=True)
        totalevents = int(params[1,1])
        truelambda = float(params[1,0])
        truelogmass = float(params[1,2])
    except:
        print(f"Could not access information from run directory = {rundirs[0]}.")
        print(f"Accessing baseline parameters and information from the next run directory = {rundirs[runaccess_startingindex]}")
        try:
            params              = np.load(f"{rundirs[runaccess_startingindex]}/params.npy")
            bkgmargresults = np.load(f'{rundirs[runaccess_startingindex]}/bkgmargresults.npy', allow_pickle=True)
            propmargresults = np.load(f'{rundirs[runaccess_startingindex]}/propmargresults.npy', allow_pickle=True)
            totalevents = int(params[1,1])
            truelambda = float(params[1,0])
            truelogmass = float(params[1,2])
            
            runaccess_startingindex+=1
        except:
            raise Exception("You have more than two runs that have no results. Aborting.")
        



    for rundir in rundirs[runaccess_startingindex:]:
        try:
            bkgmargresults    = np.concatenate((bkgmargresults,  np.load(f'{rundir}/bkgmargresults.npy', allow_pickle=True)))
            propmargresults   = np.concatenate((propmargresults, np.load(f'{rundir}/propmargresults.npy', allow_pickle=True)))
            tempparams      = np.load(f"{rundir}/params.npy")
            totalevents     += int(tempparams[1,1])
            if truelambda!=float(tempparams[1,0]):
                raise Exception("The value of lambda is not constant across your runs")
            
            if truelogmass!=float(tempparams[1,2]):
                raise Exception("The value of truelogmass is not constant across your runs")
        except:
            print(f"You are missing information from the run directory={rundir}")
        
    print(f"Total events: {totalevents}")
    print(f"True lambda val: {truelambda}")
    print(f"True logmassval: {truelogmass}")

    recyclingresults = runrecycle(propmargresults, bkgmargresults, logpropdist, sigdistsetup, log10eaxis=log10eaxis, recyclingcores = reccores, 
                                  nlive = 1000, print_progress=True)

    np.save(f'{stemdirectory}/recyclingresults.npy', recyclingresults)

    # chime.info('sonic')
        
    



    