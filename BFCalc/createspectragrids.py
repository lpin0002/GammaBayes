# -*- coding: utf-8 -*-
from scipy import integrate, interpolate
import os, sys, warnings
import numpy as np
simpson =  integrate.simps
from gammapy.astro.darkmatter import (
    PrimaryFlux,
)

from astropy.table import Table
# import matplotlib.pyplot as plt
from scipy import special, interpolate
import matplotlib.pyplot as plt
import sys

sys.path.append("..")


channel_registry = {
    "eL": "eL",
    "eR": "eR",
    "e": "e",
    "muL": r"\[Mu]L",
    "muR": r"\[Mu]R",
    "mu": r"\[Mu]",
    "tauL": r"\[Tau]L",
    "tauR": r"\[Tau]R",
    "tau": r"\[Tau]",
    "q": "q",
    "c": "c",
    "b": "b",
    "t": "t",
    "WL": "WL",
    "WT": "WT",
    "W": "W",
    "ZL": "ZL",
    "ZT": "ZT",
    "Z": "Z",
    "g": "g",
    "gamma": r"\[Gamma]",
    "h": "h",
    "nu_e": r"\[Nu]e",
    "nu_mu": r"\[Nu]\[Mu]",
    "nu_tau": r"\[Nu]\[Tau]",
    "V->e": "V->e",
    "V->mu": r"V->\[Mu]",
    "V->tau": r"V->\[Tau]",
}


# The below saves the absolute path to the folder containing __this__ file
modulefolderpath = os.path.join(os.path.dirname(__file__))

immediatespectratable = Table.read(
    modulefolderpath+'/dark_matter_spectra/AtProduction_gammas.dat',
                                   format="ascii.fast_basic",
                guess=False,
                delimiter=" ",)

# p p test
# Transient mailing list
# Convert to logx
def singlechannel_diffflux(mass, channel):
    # Mass in GeV
    
    subtable = immediatespectratable[immediatespectratable["mDM"] == mass]
    log10x= subtable["Log[10,x]"].data
    energies = (10 ** subtable["Log[10,x]"].data) * mass
    dN_dlogx = subtable[channel].data
    dN_dE = dN_dlogx / (energies * np.log(10))
    return log10x, dN_dE


# Are in GeV
massvalues  = immediatespectratable["mDM"].data
massvalues = np.unique(massvalues)
log10xvals  = immediatespectratable["Log[10,x]"].data
log10xvals = np.unique(log10xvals)


# massenergygrid = np.meshgrid(massvalues, energyvalues)
def singlechannelgrid(channel):
    massvalues  = immediatespectratable["mDM"].data
    massvalues = np.unique(massvalues)
    difffluxgrid = []
    
    for massvalue in massvalues:
        log10xvals, dN_dE = singlechannel_diffflux(massvalue, channel)
        difffluxgrid.append(dN_dE)
        
    return difffluxgrid
gridtointerpolate   = np.load(modulefolderpath+f"/griddata/channel=tau_massenergy_diffflux_grid.npy")
massvalues          = np.load(modulefolderpath+f"/griddata/massvals_massenergy_diffflux_grid.npy")
log10xvals        = np.load(modulefolderpath+f"/griddata/log10xvals_massenergy_diffflux_grid.npy")

twodinterpolationfunc =  interpolate.interp2d(np.log10(massvalues/1e3), log10xvals, np.array(gridtointerpolate).T, 
                                kind='linear', bounds_error=False, fill_value=0)
def getspectrafunc(mDM, channel):
    def onedinterpolationfunc(energy):        
        return twodinterpolationfunc(np.log10(mDM), np.log10(energy/mDM).flatten()).reshape(energy.shape)
    
    return onedinterpolationfunc







# for channel, channelstored in channel_registry.items():
#     np.save(modulefolderpath+f"/griddata/channel={channel}_massenergy_diffflux_grid.npy", singlechannelgrid(channelstored))
# np.save(modulefolderpath+f"/griddata/massvals_massenergy_diffflux_grid.npy", massvalues)
# np.save(modulefolderpath+f"/griddata/log10xvals_massenergy_diffflux_grid.npy", log10xvals)

