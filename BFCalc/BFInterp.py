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
from .creategriddata_and_check import getspectrafunc
import sys
sys.path.append("..")



# The below saves the absolute path to the folder containing __this__ file
modulefolderpath = os.path.join(os.path.dirname(__file__))




bfmlambdaarray = np.load(modulefolderpath+'/bfmlambdaarray.npy')



def DM_spectrum_setup(logmDM=-0.7, lambdainput=0.1, normeaxis=np.logspace(-6, 4, 3001)):
    """A function to generate a function that computes the log of the spectra with log energy inputs for use in probability analysis.

    The branching fractions are generated by interpolating outputs from the coding package microOmegas. From there,
    the single channel spectra are taken from the PPPC 4 DM ID. And finally, these spectra are normalized to create pdfs
    that can then be sampled.

    Args:
        lamda_input: The coupling constant between the Scalar singlet field and the higgs field
        logmDM: The log base 10 of the mass of the Scalar Singlet particle in TeV.

    Returns: dmfullsoec
        dmfullspec: (Generator) A function of the natural log of the spectra that takes in log base 10 energy values.
    """
    eaxis=np.logspace(-6, 4, 3001)
    os.environ["GAMMAPY_DATA"]   = modulefolderpath
    # Annoying scaling thing with microOMEGAS. The only unit I could give it was in GeV, but the rest of my code is
        # is TeV, so here's a basic adhoc solution.
    
    mDM = np.power(10.,logmDM)
    dataArr=bfmlambdaarray
  
    Lambda = dataArr[0,:,1]
    m_DM = dataArr[:,0,0]
    
    # print(np.log10(m_DM)-3)
    # relic = dataArr[:,:,2]
    Bfw = dataArr[:,:,3]
    Bfz = dataArr[:,:,4]
    Bfh = dataArr[:,:,5]
    Bfb = dataArr[:,:,6]
    Bfc = dataArr[:,:,7]
    Bfl = dataArr[:,:,8]
    Bfg = dataArr[:,:,9]
    Bft = dataArr[:,:,10]
    smoothing = 0.3
    k = 2
    # relicdensityinterped = interpolate.RectBivariateSpline(m_DM, Lambda, relic, s=smoothing, kx = k, ky=k)(lambdainput,mDM)[0]
    Bfw_interped = interpolate.RectBivariateSpline(m_DM, Lambda, Bfw, s=smoothing, kx = k, ky=k)(lambdainput,mDM)[0]
    Bfz_interped = interpolate.RectBivariateSpline(m_DM, Lambda, Bfz, s=smoothing, kx = k, ky=k)(lambdainput,mDM)[0]
    Bfh_interped = interpolate.RectBivariateSpline(m_DM, Lambda, Bfh, s=smoothing, kx = k, ky=k)(lambdainput,mDM)[0]
    Bfb_interped = interpolate.RectBivariateSpline(m_DM, Lambda, Bfb, s=smoothing, kx = k, ky=k)(lambdainput,mDM)[0]
    Bfc_interped = interpolate.RectBivariateSpline(m_DM, Lambda, Bfc, s=smoothing, kx = k, ky=k)(lambdainput,mDM)[0]
    Bfl_interped = interpolate.RectBivariateSpline(m_DM, Lambda, Bfl, s=smoothing, kx = k, ky=k)(lambdainput,mDM)[0]
    Bfg_interped = interpolate.RectBivariateSpline(m_DM, Lambda, Bfg, s=smoothing, kx = k, ky=k)(lambdainput,mDM)[0]
    Bft_interped = interpolate.RectBivariateSpline(m_DM, Lambda, Bft, s=smoothing, kx = k, ky=k)(lambdainput,mDM)[0]

    BFnorm = Bfw_interped+Bfz_interped+Bfh_interped+Bfb_interped+Bfc_interped+Bfl_interped+Bfg_interped+Bft_interped
    # if relicdensityinterped>0.12:
    #     warnings.warn(f"""The mass and higgs coupling constant values you have chosen, {mDM} GeV and {lambdainput} 
    #     respectively, lead to a relic density larger  than the observed value of 0.12. Thus, the output of this routine 
    #     is not physically possible. We suggest increasing the coupling value and decreasing the mass value if possible.""")

    # possiblechannels = ["W", "Z", "h", "b", "c", "tau", "g", "t"]

    # interpedchannels = [Bfw_interped,Bfz_interped,Bfh_interped,Bfb_interped,
    #                     Bfc_interped,Bfl_interped,Bfg_interped,Bft_interped]
    
    logjacob = np.log(eaxis) + np.log(np.log(10))+np.log(np.log10(eaxis)[1]-np.log10(eaxis)[0])


    # def singlechannel_spec(logmDM=logmDM, channel="W", energrange = eaxis):
    #     mDM = np.power(10., logmDM)
    #     DMfluxobj = PrimaryFlux(mDM=f"{mDM} TeV", channel=channel)
    #     DMfluxDict = DMfluxobj.table_model.to_dict().get('spectral')

    #     energies = np.array(DMfluxDict.get('energy').get('data'))/1000  # in TeV
    #     difffluxes = np.array(DMfluxDict.get('values').get('data'))*1000  # in 1/TeV
        
    #     #print(DMfluxDict.get('values').get('unit'))

    #     func = interpolate.interp1d(x=energies, y=difffluxes, assume_sorted=True,# y=fluxes/norm, assume_sorted=True,
    #                                 bounds_error=False, kind='cubic',
    #                                 fill_value=0)
    #     # The below creates a normalised pdf, but in log Energy. As that is typically what the sampling goes off of
    #     # plt.figure()
    #     # plt.title(f"{channel}, {energrange[0]}")
    #     # plt.plot(energrange, func(energrange))
    #     # plt.yscale('log')
    #     # plt.xlim([0,mDM+0.1*np.abs(mDM)])
    #     # plt.show()


    #     return func(energrange)

    def dm_fullspec(logmDM=logmDM):
        """A function that returns a function """
        mDM = 10**logmDM
        
        yvals = Bfw_interped * getspectrafunc(mDM=mDM, channel="W")(normeaxis) \
            + getspectrafunc(mDM=mDM, channel="Z")(normeaxis) * Bfz_interped \
            + getspectrafunc(mDM=mDM, channel="h")(normeaxis) * Bfh_interped \
            + getspectrafunc(mDM=mDM, channel="b")(normeaxis) * Bfb_interped \
            + getspectrafunc(mDM=mDM, channel="c")(normeaxis) * Bfc_interped \
            + getspectrafunc(mDM=mDM, channel="tau")(normeaxis) * Bfl_interped \
            + getspectrafunc(mDM=mDM, channel="g")(normeaxis) * Bfg_interped \
            + getspectrafunc(mDM=mDM, channel="t")(normeaxis) * Bft_interped
        BFnorm = Bfw_interped+Bfz_interped+Bfh_interped+Bfb_interped+Bfc_interped+Bfl_interped+Bfg_interped+Bft_interped
        log10eaxis = np.log10(normeaxis)
        yvals = np.squeeze(yvals/BFnorm)
        # whereobject = np.where(log10eaxis>=logmDM)
        # try:
        #     stopindex = whereobject[0][0]
        # except:
        #     if logmDM<log10eaxis[0]:
        #         stopindex = 2
        #     elif logmDM>log10eaxis[-1]:
        #         stopindex = -1
        #     else:
        #         raise Exception("Bounds on the index are incompatible with stopping index.")
                
        # if stopindex<2:
        #     stopindex=2
        
        yvals = np.log(yvals)#[:stopindex]
        log10eaxis = log10eaxis#[:stopindex]
                
        norm = special.logsumexp(yvals+np.log(10**log10eaxis)+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0]))

        if np.isneginf(norm) or np.isnan(norm):
            norm=0 
        
        fullspectrum = interpolate.interp1d(y=yvals-norm, x =log10eaxis, kind='linear',
                                              assume_sorted=True, bounds_error=False, fill_value=-np.inf)
        
        return fullspectrum 

    return dm_fullspec(logmDM=logmDM)

# examplespec = DM_spectrum_setup(logmDM=1.0)

# plt.figure()
# plt.plot(np.linspace(-2,2,401), examplespec(np.linspace(-2,2,401)), label=special.logsumexp(examplespec(np.linspace(-2,2,401))+np.log(np.logspace(-2,2,401))))
# plt.xlim([-2,2])
# plt.show()