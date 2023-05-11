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
import pandas as pd

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



    
# # print(immediatespectratable)

# subtable = immediatespectratable[immediatespectratable["mDM"] == immediatespectratable["mDM"][0]]
# energypast = (10 ** subtable["Log[10,x]"]) * immediatespectratable["mDM"][0]
# badindices = []
# plt.figure()


def singlechannel_diffflux(mass, channel):
    subtable = immediatespectratable[immediatespectratable["mDM"] == mass]
    energies = (10 ** subtable["Log[10,x]"]) * mass
    dN_dlogx = subtable[channel]
    dN_dE = dN_dlogx / (energies * np.log(10))
    
    return energies, dN_dE

massvalues  = immediatespectratable["mDM"].data
massvalues = np.unique(massvalues)
energyvalues = np.logspace(-6,2,1001)

# massenergygrid = np.meshgrid(massvalues, energyvalues)
def singlechannelgrid(channel):
    massvalues  = immediatespectratable["mDM"].data
    massvalues = np.unique(massvalues)
    difffluxgrid = []
    for massvalue in massvalues:
        energies, dN_dE = singlechannel_diffflux(massvalue, channel)
        massvalueinterp1d = interpolate.interp1d(x=energies/1e3, y=dN_dE, kind='quadratic', fill_value=0, bounds_error=False)
        difffluxgrid.append(massvalueinterp1d(energyvalues))
        
    return difffluxgrid

def getspectrafunc(mDM, channel):
    gridtointerpolate   = np.load(modulefolderpath+f"/griddata/channel={channel}_massenergy_diffflux_grid.npy")
    massvalues          = np.load(modulefolderpath+f"/griddata/massvals_massenergy_diffflux_grid.npy")
    energyvalues        = np.load(modulefolderpath+f"/griddata/energyvals_massenergy_diffflux_grid.npy")
    
    func =  interpolate.interp2d(massvalues/1e3, energyvalues, np.array(gridtointerpolate).T, kind='linear')
    
    return lambda energy: func(mDM, energy)








# for channel, channelstored in channel_registry.items():
#     np.save(modulefolderpath+f"/griddata/channel={channel}_massenergy_diffflux_grid.npy", singlechannelgrid(channelstored))
# np.save(modulefolderpath+f"/griddata/massvals_massenergy_diffflux_grid.npy", massvalues)
# np.save(modulefolderpath+f"/griddata/energyvals_massenergy_diffflux_grid.npy", energyvalues)

# gridtointerpolate = singlechannelgrid('W')

# print(np.array(gridtointerpolate).shape)


# Wchannelinterped = interpolate.interp2d(massvalues/1e3, energyvalues, np.array(gridtointerpolate).T, kind='linear')

# print("interped value: ", Wchannelinterped(1,1))

# os.environ["GAMMAPY_DATA"]   = modulefolderpath
# difffluxvalues = []
# massaxis = np.logspace(-3,2,251)
# for mass in massaxis:
#         DMfluxobj = PrimaryFlux(mDM=f"{mass} TeV", channel='W')
#         DMfluxDict = DMfluxobj.table_model.to_dict().get('spectral')

#         energies = np.array(DMfluxDict.get('energy').get('data'))/1000  # in TeV
#         difffluxes = np.array(DMfluxDict.get('values').get('data'))*1000
        
#         difffluxval = difffluxes[np.abs(energies-1).argmin()]
        
#         difffluxvalues.append(difffluxval)
        

        

# plt.figure()
# plt.plot(np.log10(massaxis), difffluxvalues, label='gammapy')
# plt.plot(np.log10(massaxis), Wchannelinterped(massaxis, np.logspace(-1,1,21)).T, label='Mine')
# plt.xlabel(r"$log_10(m)$")
# plt.ylabel("Differential Flux [1/TeV]")
# plt.show()



# plt.figure()
# plt.plot(energyvalues, getspectrafunc(0.1,"W")(energyvalues), label='mDM=0.1 TeV')
# plt.plot(energyvalues, getspectrafunc(1,"W")(energyvalues), label='mDM=1 TeV')
# plt.plot(energyvalues, getspectrafunc(10,"W")(energyvalues), label='mDM=10 TeV')
# plt.legend()
# plt.xlabel(r"Energy [TeV]")
# plt.loglog()
# plt.ylabel("Differential Flux [1/TeV]")
# plt.show()