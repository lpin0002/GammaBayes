# -*- coding: utf-8 -*-
from scipy import integrate, interpolate
import os, sys, warnings
import numpy as np
simpson =  integrate.simps
from gammapy.astro.darkmatter import (
    PrimaryFlux,
)
import matplotlib.pyplot as plt
from scipy import special
from .createspectragrids import getspectrafunc
import sys
sys.path.append("..")



# The below saves the absolute path to the folder containing __this__ file
modulefolderpath = os.path.join(os.path.dirname(__file__))




# bfmlambdaarray = np.load(modulefolderpath+'/bfmlambdaarray.npy')



def DM_spectrum_setup(logmDM=-0.7, normeaxis=np.logspace(-6, 4, 3001)):
    # eaxis=np.logspace(-6, 4, 3001)
    eaxis = normeaxis
    

    def dm_fullspec(logmDM=logmDM):
        """A function that returns a function """
        mDM = 10**logmDM
        
        yvals = getspectrafunc(mDM=mDM, channel="W")(eaxis)
        
        log10eaxis = np.log10(eaxis)
        
        try:
            stopindex = np.where(log10eaxis>=logmDM)
            stopindex = stopindex[0][0]
        except:
            if log10eaxis[-1]<logmDM:
                stopindex = -1
            else:
                stopindex = 2
                yvals = 0*yvals
            
        
        logyvals = np.squeeze(np.log(yvals))
        
        # To get rid of interpolation artefacts
        logyvals[stopindex:] = -np.inf

                
        norm = special.logsumexp(logyvals+np.log(10**log10eaxis)+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0]))

        if np.isneginf(norm) or np.isnan(norm):
            norm=0 
            
        # print(special.logsumexp(logyvals-norm+np.log(10**log10eaxis)+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0])))
        
        # Nearest decreases computation time an would be equivalent to linear as we are interpolating on the input axis
        fullspectrum = interpolate.interp1d(y=logyvals-norm, x =log10eaxis, kind='nearest',
                                              assume_sorted=True, bounds_error=False, fill_value=-np.inf)
        
        return fullspectrum 

    return dm_fullspec(logmDM=logmDM)

