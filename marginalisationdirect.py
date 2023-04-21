
from utils import inverse_transform_sampling, axis, bkgdist, makedist, edisp, eaxis_mod, color
from scipy import integrate, special, interpolate, stats
import numpy as np
import os, time, random, warnings, concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt
import chime
from BFCalc.BFInterp import DM_spectrum_setup

# chime.info('sonic')

sigdistsetup = makedist
# Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
np.seterr(divide='ignore', invalid='ignore')

sigsamples          = np.load("truesigsamples.npy")
sigsamples_measured = np.load("meassigsamples.npy")
bkgsamples          = np.load("truebkgsamples.npy")
bkgsamples_measured = np.load("measbkgsamples.npy")
params              = np.load("params.npy")
params[1,:]         = params[1,:]
truelogmass     = float(params[1,2])
nevents         = int(params[1,1])
truelambdaval   = float(params[1,0])
truevals            = np.concatenate((sigsamples, bkgsamples))
measuredvals        = np.concatenate((sigsamples_measured,bkgsamples_measured))


logmasslowerbound = truelogmass-1/np.sqrt(nevents*2)
if logmasslowerbound<axis[0]:
       logmasslowerbound = axis[0]

nbins = 41

logmassrange = np.linspace(logmasslowerbound,truelogmass+1/np.sqrt(nevents*2),nbins)
lambdarange = np.linspace(truelambdaval-1/np.sqrt(nevents/2),truelambdaval+1/np.sqrt(nevents/2),nbins)
np.save('logmassrange.npy',logmassrange)
np.save('lambdarange.npy',lambdarange)
# lambdarange = np.array([0.45, 0.5])

print(color.BOLD+f"""\n\n{color.BOLD}{color.GREEN}IMPORTANT PARAMETERS: {color.END}
{color.YELLOW}number of events{color.END} being analysed/were simulated is {nevents:.1e}. 

{color.YELLOW}true log mass value{color.END} used for the signal model is {truelogmass} or equivalently a mass of roughly {np.round(np.power(10., truelogmass),3):.2e}.

{color.YELLOW}fraction of signal events to total events{color.END} is {truelambdaval}.

{color.YELLOW}bounds for the log energy range{color.END} are {axis[0]:.2e} and {axis[-1]:.2e} translating into energy bounds of {np.power(10.,axis[0]):.2e} and {np.power(10.,axis[-1]):.2e}.

{color.YELLOW}bounds for the log mass range [TeV]{color.END} are {logmassrange[0]:.2e} and {logmassrange[-1]:.2e} translating into mass bounds of {np.power(10.,logmassrange[0]):.2e} and {np.power(10.,logmassrange[-1]):.2e} [TeV].

{color.YELLOW}bounds for the lambda range{color.END} are {lambdarange[0]:.2e} and {lambdarange[-1]:.2e}.

\n""")



edispnorms = np.array([special.logsumexp(edisp(axis,axisval)+eaxis_mod) for axisval in axis])

if -np.inf in edispnorms:
       print(color.BOLD+"Your energy dispersion normalisation has -np.inf inside, which will almostly definitely mean your energy dispersion or the normalisation is wrong."+color.END)

edisplist = []
bkgmarglist = []
bkgdistnormed = bkgdist(axis) - special.logsumexp(bkgdist(axis)+eaxis_mod)


print(f"There are {color.BLUE}{nevents}{color.END} events being analyzed.")
for i, sample in tqdm(enumerate(measuredvals),desc="Calculating edisp vals and bkg marginalisation", ncols=100):
        edisplist.append(edisp(sample,axis)-edispnorms)
        bkgmarglist.append(special.logsumexp(bkgdistnormed+edisplist[i]+eaxis_mod))
edisplist = np.array(edisplist)


sigmarglogzvals = []
tempsigfuncs = []


for logmass in tqdm(logmassrange, desc="Marginalising over signal priors", ncols=100):
       tempsigdist = sigdistsetup(logmass)
       tempsigfuncs.append(tempsigdist)
       tempmargval = 0
       tempmarglogmassrow = []
       tempsigdist = sigdistsetup(logmass)
       tempsigdistaxis = tempsigdist(axis) - special.logsumexp(tempsigdist(axis)+eaxis_mod)
       for i, sample in enumerate(measuredvals):
              tempsigmarg = special.logsumexp(tempsigdistaxis+edisplist[i]+eaxis_mod)
              tempmarglogmassrow.append(tempsigmarg)
       sigmarglogzvals.append(tempmarglogmassrow)


logmassposterior = []
for j in tqdm(range(len(logmassrange)), ncols=100, desc="Computing log posterior in lambda and logmDM"):
       templogmassrow = []
       for lambdaval in lambdarange:
              tempmargval = 0
              for i, sample in enumerate(measuredvals):
                     tempmargval += np.logaddexp(np.log(lambdaval)+sigmarglogzvals[j][i],np.log(1-lambdaval)+bkgmarglist[i])
              # print(f"{tempmargval:.2e}", end='\r')
              
              templogmassrow.append(tempmargval)
              
       logmassposterior.append(templogmassrow)

print("\n")
# Some values of mass can give nan results because the probability is zero everywhere (mass below energy range)
       # hence when I normalise it python returns NaN values. I'm replacing those with -np.inf which are easier 
       # to deal with
# logmassposterior = np.where(np.isnan(logmassposterior),-np.inf, logmassposterior)
normedlogposterior = logmassposterior - special.logsumexp(logmassposterior)
np.save("normedlogposteriorDirect.npy", normedlogposterior)


chime.info('sonic')







