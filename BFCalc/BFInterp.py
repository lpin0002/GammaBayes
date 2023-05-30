# -*- coding: utf-8 -*-
from scipy import interpolate
import os, sys
import numpy as np
from gammapy.astro.darkmatter import (
    PrimaryFlux,
)
from utils import makelogjacob
from scipy import special
from .createspectragrids import getspectrafunc
import sys
sys.path.append("..")



# The below saves the absolute path to the folder containing __this__ file
modulefolderpath = os.path.join(os.path.dirname(__file__))




# bfmlambdaarray = np.load(modulefolderpath+'/bfmlambdaarray.npy')



def DM_spectrum_setup(logmDM=-0.7, normeaxis=np.logspace(-6, 4, 3001)):
    # eaxis=np.logspace(np.log10(normeaxis[0]), np.log10(normeaxis[-1]), 200001)
    eaxis = normeaxis
    

    def dm_fullspec(logmDM=logmDM):
        """A function that returns a function """
        mDM = 10**logmDM
        
        logdN_dE = np.log(getspectrafunc(mDM=mDM, channel="b")(eaxis))
        logdN_dE = np.squeeze(logdN_dE)
        log10eaxis = np.log10(eaxis)
        logdN_dE = logdN_dE[log10eaxis<logmDM]
        log10eaxis = log10eaxis[log10eaxis<logmDM]

        if log10eaxis.shape[0]>1:
            logdN_dE = np.squeeze(logdN_dE)
            logjacob = makelogjacob(log10eaxis)
            norm = special.logsumexp(logdN_dE+logjacob) 

            if np.isneginf(norm) or np.isnan(norm):
                norm=0 
                
            # print(special.logsumexp(logyvals-norm+np.log(10**log10eaxis)+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0])))
            
            fullspectrum = interpolate.interp1d(y=logdN_dE-norm, x =log10eaxis, kind='linear',
                                                assume_sorted=True, bounds_error=False, fill_value=-np.inf)
        else:
            def fullspectrum(energ):
                if type(energ)==np.ndarray:
                    return np.full(energ.shape, -np.inf)
                else:
                    return -np.inf
        
        return fullspectrum 

    return dm_fullspec(logmDM=logmDM)

