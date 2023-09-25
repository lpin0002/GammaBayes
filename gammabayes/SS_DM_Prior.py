import numpy as np
from astropy import units as u
from gammapy.astro.darkmatter import (
    profiles,
    JFactory
)
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
from scipy import interpolate
import pandas as pd

from os import path
BFCalc_dir = path.join(path.dirname(__file__), 'BFCalc')

class SS_DM_dist(object):
    
    def __init__(self, longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile()):
        self.longitudeaxis = longitudeaxis
        self.latitudeaxis = latitudeaxis
        self.density_profile = density_profile
        
    
    
        darkSUSY_to_PPPC_converter = {
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


        darkSUSY_BFs_cleaned = pd.read_csv(BFCalc_dir+'/darkSUSY_BFs/darkSUSY_BFs_cleaned.csv', delimiter=' ')

        darkSUSY_massvalues = darkSUSY_BFs_cleaned.iloc[:,1]/1e3

        darkSUSY_lambdavalues = darkSUSY_BFs_cleaned.iloc[:,2]

        channelfuncdictionary = {}

       
        log10xvals = np.load(BFCalc_dir+f"/griddata/log10xvals_massenergy_diffflux_grid.npy")
        massvalues = np.load(BFCalc_dir+f"/griddata/massvals_massenergy_diffflux_grid.npy")

        for darkSUSYchannel in list(darkSUSY_to_PPPC_converter.keys()):
            try:
                gammapychannel = darkSUSY_to_PPPC_converter[darkSUSYchannel]
                
                tempspectragrid = np.load(BFCalc_dir+f"/griddata/channel={gammapychannel}_massenergy_diffflux_grid.npy")
                
                channelfuncdictionary[darkSUSYchannel] = interpolate.RegularGridInterpolator((np.log10(massvalues/1e3), log10xvals), np.array(tempspectragrid), 
                                                                                        method='linear', bounds_error=False, fill_value=1e-3000)
            except:
                channelfuncdictionary[darkSUSYchannel] = lambda logmass, log10x: log10x*0

        self.channelfuncdictionary = channelfuncdictionary
        self.partial_sigmav_interpolator_dictionary = {channel: interpolate.LinearNDInterpolator((darkSUSY_massvalues, darkSUSY_lambdavalues),darkSUSY_BFs_cleaned.iloc[:,idx+3]) for idx, channel in enumerate(list(darkSUSY_to_PPPC_converter.keys()))}
        
        self.profile = density_profile

        # Adopt standard values used in HESS
        profiles.DMProfile.DISTANCE_GC = 8.5 * u.kpc
        profiles.DMProfile.LOCAL_DENSITY = 0.39 * u.Unit("GeV / cm3")

        self.profile.scale_to_local_density()

        self.central_position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
        self.geom = WcsGeom.create(skydir=self.central_position, 
                            binsz=(self.longitudeaxis[1]-self.longitudeaxis[0], self.latitudeaxis[1]-self.latitudeaxis[0]),
                            width=(self.longitudeaxis[-1]-self.longitudeaxis[0]+self.longitudeaxis[1]-self.longitudeaxis[0], self.latitudeaxis[-1]-self.latitudeaxis[0]+self.latitudeaxis[1]-self.latitudeaxis[0]),
                            frame="galactic")


        jfactory = JFactory(
            geom=self.geom, profile=self.profile, distance=profiles.DMProfile.DISTANCE_GC
        )
        self.diffjfact_array = (jfactory.compute_differential_jfactor().value).T

        self.diffJfactor_function = interpolate.RegularGridInterpolator((self.longitudeaxis, self.latitudeaxis), self.diffjfact_array, method='linear', bounds_error=False, fill_value=0)


    def nontrivial_coupling(self, logmass, logenergy, coupling=0.1, partial_sigmav_interpolator_dictionary=None, channelfuncdictionary=None):
        if partial_sigmav_interpolator_dictionary is None:
            partial_sigmav_interpolator_dictionary = self.partial_sigmav_interpolator_dictionary
            
        if channelfuncdictionary is None:
            channelfuncdictionary = self.channelfuncdictionary
        
        logspectra = -np.inf
        for channel in channelfuncdictionary.keys():
            logspectra = np.logaddexp(logspectra, np.log(partial_sigmav_interpolator_dictionary[channel](10**logmass, coupling)*channelfuncdictionary[channel]((logmass, logenergy-logmass))))
            
        return logspectra
    

    
    
    
    
    
    def func_setup(self):
    
        def DM_signal_dist(log10eval, lonval, latval, logmass, coupling=0.1):
            # try:
            spectralvals = self.nontrivial_coupling(logmass.flatten(), log10eval.flatten()).reshape(log10eval.shape)
            # except:
            #     spectralvals = np.squeeze(logspecfunc(logmass, log10eval))
            
            spatialvals = np.log(self.diffJfactor_function((lonval.flatten(), latval.flatten()))).reshape(log10eval.shape)
                    
            logpdfvalues = spectralvals+spatialvals
            
            return logpdfvalues
        
        return DM_signal_dist
    
    