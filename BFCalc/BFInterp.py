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
    spectralfunc = getspectrafunc(mDM=10**logmDM, channel="tau")
    def dm_fullspec(logenerg):
        return np.log(spectralfunc(10**logenerg)) 
            
    return dm_fullspec

