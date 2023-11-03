
from ..likelihoods.IRFs import bkgfull
import numpy as np

bkgfull2d = bkgfull.to_2d()
bkgfull2doffsetaxis = bkgfull2d.axes['offset'].center.value
offsetaxisresolution = bkgfull2doffsetaxis[1]-bkgfull2doffsetaxis[0] # Comes out to 0.2
latbound            = 3.
lonbound            = 3.5



latitudeaxis            = np.linspace(-latbound, latbound, int(round(2*latbound/0.4)))
latitudeaxistrue        = np.linspace(-latbound, latbound, int(round(2*latbound/0.2)))

longitudeaxis           = np.linspace(-lonbound, lonbound, int(round(2*lonbound/0.4))) 
longitudeaxistrue       = np.linspace(-lonbound, lonbound, int(round(2*lonbound/0.2))) 


# Restricting energy axis to values that could have non-zero or noisy energy dispersion (psf for energy) values
log10estart             = -1
log10eend               = 2
log10erange             = log10eend - log10estart
log10eaxis              = np.linspace(log10estart,log10eend,int(np.round(log10erange*50))+1)
log10eaxistrue          = np.linspace(log10estart,log10eend,int(np.round(log10erange*200))+1)



def makelogjacob(log10eaxis=log10eaxis):
    """_summary_

    Args:
        log10eaxis (np.ndarray, optional): Axis of discrete values of log10 energy values. 
        Defaults to log10eaxis.

    Returns:
        np.ndarray: Log jacobian for using log10 energy to get integral over energy
    """
    outputlogjacob = np.log(10**log10eaxis)#+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0])
    return outputlogjacob

logjacob = makelogjacob(log10eaxis)
logjacobtrue = makelogjacob(log10eaxistrue)



def create_axis(lower_bound, upper_bound, number_of_bins_per_unit=100, resolution=None):
    if resolution is None:
        return np.linspace(lower_bound, 
                           upper_bound, 
                           int(round(number_of_bins_per_unit*(upper_bound-lower_bound))+1))
    else:
        return np.linspace(lower_bound, 
                         upper_bound, 
                         int(round((upper_bound-lower_bound)/resolution)+1))
    
def create_axes(log10_energy_min, log10_energy_max, 
                     log10_energy_bins_per_decade, spatial_res, 
                     longitude_min, longitude_max,
                     latitude_min, latitude_max):
    
    print(log10_energy_min, log10_energy_max, 
                     log10_energy_bins_per_decade, spatial_res, 
                     longitude_min, longitude_max,
                     latitude_min, latitude_max)
    
    log10_eaxis = create_axis(log10_energy_min, log10_energy_max, number_of_bins_per_unit=log10_energy_bins_per_decade)
    longitude_axis = create_axis(longitude_min, longitude_max, resolution=spatial_res)
    latitude_axis = create_axis(latitude_min, latitude_max, resolution=spatial_res)

    return log10_eaxis, longitude_axis, latitude_axis