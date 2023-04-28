
import os, sys, numpy as np, time
from scipy import special, stats
from tqdm import tqdm
from runrecycle import runrecycle
from utils import logpropdist, makedist
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
    runnum=1
    print("runnum: ", runnum)
    params              = np.load(f"{rundirs[0]}/params.npy")
    bkgmargresults = np.load(f'{rundirs[0]}/bkgmargresults.npy', allow_pickle=True)
    propmargresults = np.load(f'{rundirs[0]}/propmargresults.npy', allow_pickle=True)
    totalevents = int(params[1,1])
    truelambda = float(params[1,0])
    truelogmass = float(params[1,2])



    for rundir in rundirs[1:]:
        bkgmargresults    = np.concatenate((bkgmargresults,  np.load(f'{rundir}/bkgmargresults.npy', allow_pickle=True)))
        propmargresults   = np.concatenate((propmargresults, np.load(f'{rundir}/propmargresults.npy', allow_pickle=True)))
        tempparams      = np.load(f"{rundir}/params.npy")
        totalevents     += int(tempparams[1,1])
        if truelambda!=float(tempparams[1,0]):
            raise Exception("The value of lambda is not constant across your runs")
        
        if truelogmass!=float(tempparams[1,2]):
            raise Exception("The value of truelogmass is not constant across your runs")
        
    print(f"Total events: {totalevents}")
    print(f"True lambda val: {truelambda}")
    print(f"True logmassval: {truelogmass}")

    recyclingresults = runrecycle(propmargresults, bkgmargresults, logpropdist, sigdistsetup, recyclingcores = reccores, 
                                  nlive = 200, print_progress=True)

    np.save(f'{stemdirectory}/recyclingresults.npy', recyclingresults)

    # chime.info('sonic')
        
    



    