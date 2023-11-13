
from gammabayes.priors.CCR_bkg import log_bkg_CCR_dist, bkgfull
import numpy as np

bkgfull2d = bkgfull.to_2d()
bkgfull2doffsetaxis = bkgfull2d.axes['offset'].center.value
offsetaxisresolution = bkgfull2doffsetaxis[1]-bkgfull2doffsetaxis[0] # Comes out to 0.2
latbound            = 3.
lonbound            = 3.5



latitudeaxis            = np.linspace(-latbound, latbound, int(round(2*latbound/0.2)))
latitudeaxistrue        = np.linspace(-latbound, latbound, int(round(2*latbound/0.1)))

longitudeaxis           = np.linspace(-lonbound, lonbound, int(round(2*lonbound/0.2))) 
longitudeaxistrue       = np.linspace(-lonbound, lonbound, int(round(2*lonbound/0.1))) 


# Restricting energy axis to values that could have non-zero or noisy energy dispersion (psf for energy) values
log10estart                     = -1
log10eend                       = 2
true_energy_bins_per_decade     = 50
recon_energy_bins_per_decade    = 200
log10erange                     = log10eend - log10estart
energy_recon_axis               = np.logspace(log10estart,log10eend,int(np.round(log10erange*recon_energy_bins_per_decade))+1)
energy_true_axis                = np.logspace(log10estart,log10eend,int(np.round(log10erange*true_energy_bins_per_decade))+1)



def makelogjacob(energyaxis):
    """Generates log jacobian for using log-spaced energy for proper integrals

    Args:
        log10eaxis (np.ndarray, optional): Axis of discrete values of energy values. 

    Returns:
        np.ndarray: Log jacobian for using log-spaced energy for proper integrals
    """
    outputlogjacob = np.log(energyaxis)
    return outputlogjacob

logjacob = makelogjacob(energy_recon_axis)
logjacobtrue = makelogjacob(energy_true_axis)



def create_axis(lower_bound, upper_bound, number_of_bins_per_unit=100, resolution=None):
    if resolution is None:
        return np.linspace(lower_bound, 
                           upper_bound, 
                           int(round(number_of_bins_per_unit*(upper_bound-lower_bound))+1))
    else:
        return np.linspace(lower_bound, 
                         upper_bound, 
                         int(round((upper_bound-lower_bound)/resolution)+1))
    
def create_axes(energy_min, energy_max, 
                    energy_bins_per_decade, spatial_res, 
                    longitude_min, longitude_max,
                    latitude_min, latitude_max):
    
    print(energy_min, energy_max, 
                    energy_bins_per_decade, spatial_res, 
                    longitude_min, longitude_max,
                    latitude_min, latitude_max)
    
    energy_axis = create_axis(energy_min, energy_max, number_of_bins_per_unit=energy_bins_per_decade)
    longitude_axis = create_axis(longitude_min, longitude_max, resolution=spatial_res)
    latitude_axis = create_axis(latitude_min, latitude_max, resolution=spatial_res)

    return energy_axis, longitude_axis, latitude_axis