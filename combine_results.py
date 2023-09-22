
import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
import sys, os

try:
    stemfolder = f'data/{sys.argv[1]}'
except:
    raise Exception('The identifier you have input is causing an error.')

try:
    num_xi = int(sys.argv[2])
except:
    print("Default number of signal fraction values, 101, is chosen.")
    num_xi = 101





currentdirecyory = os.getcwd()
stemdirectory = f'{currentdirecyory}/{stemfolder}/singlerundata'
print("\nstem directory: ", stemdirectory, '\n')

rundirs = [x[0] for x in os.walk(stemdirectory)][1:]


    
margresultsarray = np.load(f'{rundirs[0]}/margresultsarray.npy', allow_pickle=True)
params = np.load(f'{rundirs[0]}/params.npy', allow_pickle=True).item()
print(params['Nevents'])
Nevents = params['Nevents']
true_xi = params['true_xi']
truelogmass = params['true_log10_mass']


logmassrange = np.load(f'{rundirs[0]}/logmassrange.npy', allow_pickle=True)

for rundir in rundirs[1:]:
    margresultsarray = np.append(margresultsarray, np.load(f'{rundir}/margresultsarray.npy', allow_pickle=True), axis=0)
    Nevents += np.load(f'{rundir}/params.npy', allow_pickle=True).item()['Nevents']


print(margresultsarray.shape)

sigmargresults = np.squeeze(np.vstack(margresultsarray[:,0])).T
bkgmargresults = np.squeeze(np.vstack(margresultsarray[:,1]))
sigmargresults.shape

#   [markdown]
# ## Calculating posterior

#  

xi_windowwidth      = 8/np.sqrt(Nevents)


xi_lowerbound       = true_xi-xi_windowwidth
xi_upperbound       = true_xi+xi_windowwidth



if xi_lowerbound<0:
    xi_lowerbound = 0
if xi_upperbound>1:
    xi_upperbound = 1


xi_range            = np.linspace(xi_lowerbound, xi_upperbound, num_xi) 

log_posterior = []

for xi_val in tqdm(xi_range, total=xi_range.shape[0], desc='combining results and generating posterior'):
    log_posterior.append([np.sum(np.logaddexp(np.log(xi_val)+sigmargresults[logmassindex,:], np.log(1-xi_val)+bkgmargresults)) for logmassindex in range(len(list(logmassrange)))])

log_posterior = np.array(log_posterior)

np.save(f'{stemfolder}/log_posterior', log_posterior)
np.save(f'{stemfolder}/xi_range', xi_range)
np.save(f'{stemfolder}/logmassrange', logmassrange)