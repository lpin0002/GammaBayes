
import sys, numpy as np, time
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
         integrationtype = ""

    sigsamples          = np.load(f"data/{identifier}/truesigsamples.npy")
    sigsamples_measured = np.load(f"data/{identifier}/meassigsamples.npy")
    bkgsamples          = np.load(f"data/{identifier}/truebkgsamples.npy")
    bkgsamples_measured = np.load(f"data/{identifier}/measbkgsamples.npy")
    params              = np.load(f"data/{identifier}/params.npy")
    params[1,:]         = params[1,:]
    truelogmass     = float(params[1,2])
    nevents         = int(params[1,1])
    truelambdaval   = float(params[1,0])
    truevals            = np.concatenate((sigsamples, bkgsamples))
    measuredvals        = np.concatenate((sigsamples_measured,bkgsamples_measured))

    logmassrange = np.load(f'data/{identifier}/logmassrange{integrationtype}.npy')
    lambdarange = np.load(f'data/{identifier}/lambdarange{integrationtype}.npy')
    edisplist = np.load(f'data/{identifier}/edisplist{integrationtype}.npy')
    bkgmarglist = np.load(f'data/{identifier}/bkgmarglist{integrationtype}.npy')
    sigmarglogzvals = np.load(f'data/{identifier}/sigmarglogzvals{integrationtype}.npy')

    logmassposterior = []
    for j in tqdm(range(len(logmassrange)), ncols=100, desc="Computing log posterior in lambda and logmDM"):
            # templogmassrow = []
            # for lambdaval in lambdarange:
            #        tempmargval = np.sum(np.logaddexp(np.log(lambdaval)+sigmarglogzvals[j],np.log(1-lambdaval)+bkgmarglist))
                    
            #        templogmassrow.append(tempmargval)
                    
            # logmassposterior.append(templogmassrow)
            templogmassrow = np.sum(np.logaddexp(np.matrix(np.log(lambdarange))+np.matrix(sigmarglogzvals[j]).T,np.matrix(np.log(1-lambdarange))+np.matrix(bkgmarglist).T),axis=0)
            templogmassrow = list(np.concatenate(np.array(templogmassrow.T)))
            logmassposterior.append(templogmassrow)

    print("\n")
    normedlogposterior = logmassposterior - special.logsumexp(logmassposterior)
    np.save(f"data/{identifier}/normedlogposterior{integrationtype}.npy", normedlogposterior)