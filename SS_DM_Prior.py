import numpy as np
from astropy import units as u
from gammapy.astro.darkmatter import (
    profiles,
    JFactory
)
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
from scipy import interpolate

def SS_DM_dist_setup(logspecfunc, longitudeaxis, latitudeaxis):
    profile = profiles.EinastoProfile()

    # Adopt standard values used in HESS
    profiles.DMProfile.DISTANCE_GC = 8.5 * u.kpc
    profiles.DMProfile.LOCAL_DENSITY = 0.39 * u.Unit("GeV / cm3")

    profile.scale_to_local_density()

    position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
    geom = WcsGeom.create(skydir=position, 
                        binsz=(longitudeaxis[1]-longitudeaxis[0], latitudeaxis[1]-latitudeaxis[0]),
                        width=(longitudeaxis[-1]-longitudeaxis[0]+longitudeaxis[1]-longitudeaxis[0], latitudeaxis[-1]-latitudeaxis[0]+latitudeaxis[1]-latitudeaxis[0]),
                        frame="galactic")


    jfactory = JFactory(
        geom=geom, profile=profile, distance=profiles.DMProfile.DISTANCE_GC
    )
    diffjfact = (jfactory.compute_differential_jfactor().value).T

    diffJfactor_function = interpolate.RegularGridInterpolator((longitudeaxis, latitudeaxis), diffjfact, method='linear', bounds_error=False, fill_value=0)
    
    def DM_signal_dist(log10eval, lonval, latval, logmass, coupling=0.1):
        try:
            spectralvals = np.squeeze(logspecfunc(logmass.flatten(), log10eval.flatten()).reshape(log10eval.shape))
        except:
            spectralvals = np.squeeze(logspecfunc(logmass, log10eval))

        spatialvals = np.squeeze(np.log(diffJfactor_function((lonval, latval))))
        
        logpdfvalues = spectralvals+spatialvals
        
        return logpdfvalues
    
    return DM_signal_dist