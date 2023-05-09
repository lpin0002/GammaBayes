
import os, sys, numpy as np, time
from scipy import special
from tqdm import tqdm

if __name__ == '__main__':
    try:
        identifier = sys.argv[1]
    except:
        identifier = time.strftime("%d%m%H")
    try:
         integrationtype = sys.argv[2]
         integrationtype = "_"+integrationtype
    except:
         integrationtype = "direct"


    currentdirecyory = os.getcwd()
    stemdirectory = currentdirecyory+f'/data/{identifier}'
    print(stemdirectory)

    rundirs = [x[0] for x in os.walk(stemdirectory)][1:]

    print(rundirs)
    print(len(rundirs))
    runnum=1
    print("runnum: ", runnum)
    params              = np.load(f"data/{identifier}/{runnum}/params.npy")
    logmassrange = np.load(f'data/{identifier}/{runnum}/logmassrange{integrationtype}.npy')
    lambdarange = np.load(f'data/{identifier}/{runnum}/lambdarange{integrationtype}.npy')
    edisplist = np.load(f'data/{identifier}/{runnum}/edisplist{integrationtype}.npy')
    bkgmarglist = np.load(f'data/{identifier}/{runnum}/bkgmarglist{integrationtype}.npy')
    sigmarglogzvals = np.load(f'data/{identifier}/{runnum}/sigmarglogzvals{integrationtype}.npy')

    dontincludenums = []
    for runnum in range(2,len(rundirs)+1):
        if not(runnum in dontincludenums):
            print("runnum: ", runnum)
            params              = np.load(f"data/{identifier}/{runnum}/params.npy")
            logmassrange = np.load(f'data/{identifier}/{runnum}/logmassrange{integrationtype}.npy')
            lambdarange = np.load(f'data/{identifier}/{runnum}/lambdarange{integrationtype}.npy')
            edisplist = np.concatenate((edisplist, np.load(f'data/{identifier}/{runnum}/edisplist{integrationtype}.npy')))
            bkgmarglist = np.concatenate((bkgmarglist, np.load(f'data/{identifier}/{runnum}/bkgmarglist{integrationtype}.npy')))
            sigmarglogzvals = np.concatenate((sigmarglogzvals.T, np.load(f'data/{identifier}/{runnum}/sigmarglogzvals{integrationtype}.npy').T)).T

        # return logmassrange, lambdarange, edisplist, bkgmarglist, sigmarglogzvals
    logmassposterior = []
    print(logmassrange.shape)
    for j in tqdm(range(logmassrange.shape[0]), ncols=100, desc="Computing log posterior in lambda and logmDM"):
            templogmassrow = np.sum(np.logaddexp(np.matrix(np.log(lambdarange))+np.matrix(sigmarglogzvals[j]).T,np.matrix(np.log(1-lambdarange))+np.matrix(bkgmarglist).T),axis=0)
            templogmassrow = list(np.concatenate(np.array(templogmassrow.T)))
            logmassposterior.append(templogmassrow)

    print("\n")
    normedlogposterior = logmassposterior - special.logsumexp(logmassposterior)
    np.save(f"data/{identifier}/normedlogposterior{integrationtype}.npy", normedlogposterior)