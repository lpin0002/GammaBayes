# -*- coding: utf-8 -*-
from scipy import integrate, interpolate
import os, sys, warnings
import numpy as np
simpson =  integrate.simps
from gammapy.astro.darkmatter import (
    PrimaryFlux,
)

import pandas as pd

from astropy.table import Table
# import matplotlib.pyplot as plt
from scipy import special, interpolate
import sys

sys.path.append("..")
np.seterr(divide = 'ignore')

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


darkSUSY_to_Gammapy_converter = {
    "nuenue":"nu_e",
    "e+e-": "e",
    "numunumu":"nu_mu",
    "mu+mu-":"mu",
    'nutaunutau':"nu_tau",
    "tau+tau-":"tau",
    "cc": "c",
    "bb": "b",
    "tt": "t",
    "W+W-": "W",
    "ZZ": "Z",
    "gg": "g",
    "gammagamma": "gamma",
    "HH": "h",
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
                                kind='linear', bounds_error=False, fill_value=1e-3000)

def getspectrafunc(mDM, channel):
    def onedinterpolationfunc(energy):        
        return twodinterpolationfunc(np.log10(mDM), np.log10(energy/mDM))
    
    return onedinterpolationfunc

def darkmatterdoubleinput(logmDM, logenergy): 
    
    return np.log(twodinterpolationfunc(logmDM, logenergy-logmDM))


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

channelnames = list(channel_registry.keys())



channeldictionary = {}
channelfuncdictionary = {}
massvalues          = np.load(modulefolderpath+f"/griddata/massvals_massenergy_diffflux_grid.npy")/1e3
log10xvals        = np.load(modulefolderpath+f"/griddata/log10xvals_massenergy_diffflux_grid.npy")

for channelname in channelnames:
    tempspectragrid = np.load(modulefolderpath+f"/griddata/channel={channelname}_massenergy_diffflux_grid.npy")
    channeldictionary[channelname] = tempspectragrid
    channelfuncdictionary[channelname] = interpolate.interp2d(np.log10(massvalues), log10xvals, np.array(tempspectragrid).T, 
                                kind='linear', bounds_error=False, fill_value=1e-3000)



bfmlambdaarray = np.load(modulefolderpath+"/temp/bfmlambdaarray.npy")
lambdarange = np.load(modulefolderpath+"/temp/lambdarange.npy")
massrange = np.load(modulefolderpath+"/temp/massrange.npy")


dataArr=bfmlambdaarray
  
  
Lambda = dataArr[0,:,1]
log_m_DM = np.log10(dataArr[:,0,0])
relic = dataArr[:,:,2]
Bfw = dataArr[:,:,3]
Bfz = dataArr[:,:,4]
Bfh = dataArr[:,:,5]
Bfb = dataArr[:,:,6]
Bfc = dataArr[:,:,7]
Bfl = dataArr[:,:,8]
Bfg = dataArr[:,:,9]
Bft = dataArr[:,:,10]

# relicdensityinterped = interpolate.interp2d(Lambda,m_DM, relic)
Bfw_interp = interpolate.interp2d(Lambda,log_m_DM, Bfw)
Bfz_interp = interpolate.interp2d(Lambda,log_m_DM, Bfz)
Bfh_interp = interpolate.interp2d(Lambda, log_m_DM, Bfh)
Bfb_interp = interpolate.interp2d(Lambda, log_m_DM, Bfb)
Bfc_interp = interpolate.interp2d(Lambda, log_m_DM, Bfc)
Bfl_interp = interpolate.interp2d(Lambda, log_m_DM, Bfl)
Bfg_interp = interpolate.interp2d(Lambda, log_m_DM, Bfg)
Bft_interp = interpolate.interp2d(Lambda, log_m_DM, Bft)

bfmainchannelnames = ['W','Z','h','b','c','tau','g','t']
branchingfractionfuncs = [Bfw_interp, Bfz_interp, Bfh_interp, Bfb_interp, Bfc_interp, Bfl_interp, Bfg_interp, Bft_interp]
branchingfunctiondictionary = dict(zip(bfmainchannelnames, branchingfractionfuncs))

def energymassinputspectralfunc(logmass, logenergy):
    finalresult = 0
    
    for channel in bfmainchannelnames:
        branchingfraction = branchingfunctiondictionary[channel](0.1,logmass)
        singlechannelspectra = channelfuncdictionary[channel](logmass, logenergy-logmass)
        finalresult+=branchingfraction*singlechannelspectra
    
    return np.log(finalresult)

def fullsinputspectralfunc(mass, higgs_coupling, energy):
    log_mass = np.log10(mass)
    log_higgs_coupling = np.log10(higgs_coupling)
    log_energy = np.log10(energy)
    finalresult = 0
    
    for channel in bfmainchannelnames:
        branchingfraction = branchingfunctiondictionary[channel](log_higgs_coupling,log_mass)
        singlechannelspectra = channelfuncdictionary[channel](log_mass, log_energy-log_mass)
        finalresult+=branchingfraction*singlechannelspectra
    
    return np.log(finalresult)



# for channel, channelstored in channel_registry.items():
#     np.save(modulefolderpath+f"/griddata/channel={channel}_massenergy_diffflux_grid.npy", singlechannelgrid(channelstored))
# np.save(modulefolderpath+f"/griddata/massvals_massenergy_diffflux_grid.npy", massvalues)
# np.save(modulefolderpath+f"/griddata/log10xvals_massenergy_diffflux_grid.npy", log10xvals)
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################
#######################################################################################################################################



# darkSUSY_BFs_cleaned = pd.read_csv('darkSUSY_BFs/darkSUSY_BFs_cleaned.csv')


# darkSUSY_massvalues = np.unique(darkSUSY_BFs_cleaned.iloc[:,0])
# darkSUSY_lambdavalues = np.unique(darkSUSY_BFs_cleaned.iloc[:,1])

# channelfuncdictionary = {}

# for darkSUSYchannel in list(darkSUSY_to_Gammapy_converter.keys()):
#     try:
#         gammapychannel = darkSUSY_to_Gammapy_converter[darkSUSYchannel]
#         tempspectragrid = np.load(modulefolderpath+f"/griddata/channel={gammapychannel}_massenergy_diffflux_grid.npy")
#         channelfuncdictionary[darkSUSYchannel] = interpolate.interp2d(np.log10(massvalues), log10xvals, np.array(tempspectragrid).T, 
#                                     kind='linear', bounds_error=False, fill_value=1e-3000)
#     except:
#         channelfuncdictionary[darkSUSYchannel] = lambda logmass, log10x: log10x*0


# partial_sigmav_interpolator_dictionary = {channel: interpolate.LinearNDInterpolator((darkSUSY_massvalues, darkSUSY_lambdavalues),darkSUSY_BFs_cleaned.iloc[:,idx+2]) for idx, channel in enumerate(list(darkSUSY_to_Gammapy_converter.keys())[2:])}

