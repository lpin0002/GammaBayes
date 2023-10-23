
import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
import sys, os, yaml
from gammabayes.utils.config_utils import read_config_file







inputs = read_config_file(sys.argv[1])


currentdirectory = os.getcwd()
stemdirectory = f"{currentdirectory}/data/{inputs['identifier']}/singlerundata"
print("\nstem directory: ", stemdirectory, '\n')

rundirs = [x[0] for x in os.walk(stemdirectory)][1:]


    
margresultsarray = np.load(f'{rundirs[0]}/margresultsarray.npy', allow_pickle=True)



logmassrange = np.load(f'{rundirs[0]}/logmassrange.npy', allow_pickle=True)

for rundir in rundirs[1:]:
    try:
        margresultsarray = np.append(margresultsarray, np.load(f'{rundir}/margresultsarray.npy', allow_pickle=True), axis=0)
    except:
        pass


print(margresultsarray.shape)

sigmargresults = np.squeeze(np.vstack(margresultsarray[:,0])).T
bkgmargresults = np.squeeze(np.vstack(margresultsarray[:,1]))
sigmargresults.shape

#   [markdown]
# ## Calculating posterior

#  

xi_windowwidth      = 8/np.sqrt(inputs['totalevents'])


xi_lowerbound       = inputs['xi']-xi_windowwidth
xi_upperbound       = inputs['xi']+xi_windowwidth



if xi_lowerbound<0:
    xi_lowerbound = 0
if xi_upperbound>1:
    xi_upperbound = 1


xi_range            = np.linspace(xi_lowerbound, xi_upperbound, inputs['nbins_xi']) 

log_posterior = []

for xi_val in tqdm(xi_range, total=xi_range.shape[0], desc='combining results and generating posterior'):
    log_posterior.append([np.sum(np.logaddexp(np.log(xi_val)+sigmargresults[logmassindex,:], np.log(1-xi_val)+bkgmargresults)) for logmassindex in range(len(list(logmassrange)))])

log_posterior = np.array(log_posterior)

np.save(f"data/{inputs['identifier']}/log_posterior", log_posterior)
np.save(f"data/{inputs['identifier']}/xi_range", xi_range)
np.save(f"data/{inputs['identifier']}/logmassrange", logmassrange)