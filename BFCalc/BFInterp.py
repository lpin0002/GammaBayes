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
    eaxis = normeaxis
    def dm_fullspec(logenerg):
        log10eaxis = np.log10(eaxis)
        logjacob = makelogjacob(log10eaxis)
        
        spectralfunc = getspectrafunc(mDM=10**logmDM, channel="tau")
        
        logdN_dE_fullaxis = np.squeeze(np.log(spectralfunc(eaxis)))
        
        normfactor = special.logsumexp(logdN_dE_fullaxis[log10eaxis<logmDM]+logjacob[log10eaxis<logmDM])         
        
        if type(logenerg)==np.ndarray:
            result = np.empty(logenerg.shape)            
            # This step is using the output of the spectrum for values below the logmass 
            #   and then setting all values above to 0 probability essentially 
            #   as annihilation shouldn't create particles heavier than the original annihilation pair
            energbelowmass = 10**logenerg[logenerg<logmDM]
            
            
            
            try:
                logvals = np.squeeze(np.log(spectralfunc(energbelowmass)))
                np.putmask(result, logenerg<logmDM ,logvals)
            except:
                if list(energbelowmass)==[]:
                    pass
                else:
                    raise Exception("There is a problem with your normalised spectra")
            
            
            np.putmask(result, logenerg>=logmDM ,-np.inf)
            
            
                      
            return np.squeeze(result-normfactor)

        else:
            if logenerg<logmDM:
                return np.log(spectralfunc(10**logenerg))-normfactor
            else:
                return -np.inf
        
    return dm_fullspec

