# Will contain a class for handling effective area, 
#   observation time and multi-observation bin overlaps
#   i.e. Source Flux Exposures

import numpy as np
from .binning_geometry import GammaBinning
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator
import copy

def trivial_log_aeff(self, energy, lon, lat, pointing_dir=None):
    return energy*0.

class GammaLogExposure:
    def __init__(self, 
                 binning_geometry:GammaBinning, 
                 irfs=None, 
                 log_exposure_map: np.ndarray = None,
                 pointing_dir: np.ndarray = None,
                 observation_times: float | np.ndarray=None, 
                 observation_times_unit: u.Unit = None,
                 use_log_aeff: bool = False
                 ):
        
        # I commonly use None for default arguments and then follow up with this
            # so that one doesn't have to remember default arguments to allow them
            # to be used in other functions/classes
        self.use_log_aeff = use_log_aeff

        if isinstance(log_exposure_map, GammaLogExposure):
            self = copy.deepcopy(log_exposure_map)
        else:
            if observation_times is None:
                observation_times = 1.*u.s
            if observation_times_unit is None:
                observation_times_unit = u.s

            self.irfs = irfs
            self.pointing_dir = pointing_dir

            if not(self.pointing_dir is None) and (self.irfs is not None):
                self.irfs.pointing_dir = self.pointing_dir


            if self.irfs is not None:
                self.log_aeff = self.irfs.log_aeff
                self.aeff_units = self.irfs.aeff_units
            else:
                self.log_aeff = trivial_log_aeff
                self.aeff_units = 1.

            self.binning_geometry = binning_geometry
            
            if hasattr(observation_times, 'unit'):
                self.time_unit = observation_times.unit
            else:
                self.time_unit = observation_times_unit


            self.unit = self.time_unit*self.aeff_units



            if isinstance(observation_times, np.ndarray):
                self.observation_times = observation_times
            elif isinstance(observation_times, (int, float, np.float64, np.float32, u.Quantity)):
                self.observation_times = np.ones(shape=self.binning_geometry.axes_dim)*observation_times
            else:
                raise ValueError(" 'observation_times' must either be an array, float, or astropy quantity")


            self.pointing_dir = pointing_dir

            if not(log_exposure_map is None):
                self.log_exposure_map = log_exposure_map
                self.exp_interpolator = RegularGridInterpolator(self.binning_geometry.axes, np.exp(self.log_exposure_map))
            else:
                self.refresh()


    def __call__(self, *args, **kwargs):
        return np.log(self.exp_interpolator(*args, **kwargs))


    # Support for indexing like a list or array
    def __getitem__(self, key: int | slice):

        return self.log_exposure_map[key]

    def __add__(self, other):

        if isinstance(other, GammaLogExposure):

            if self.binning_geometry!=other.binning_geometry:
                raise NotImplemented("Adding exposures for different binning geometries is currently not supported.")

            new_exposure_map = np.logaddexp(self.log_exposure_map, other.log_exposure_map)

            return GammaLogExposure(binning_geometry=self.binning_geometry, log_exposure_map=new_exposure_map)
        else:
            return other+self.log_exposure_map
        



    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
        

    def refresh(self):

        if self.use_log_aeff:
            log_exposure_vals = self.log_aeff(*self.binning_geometry.axes_mesh)
        else:
            log_exposure_vals = 0.
        
        log_exposure_vals+=np.log(self.observation_times.to(self.time_unit).value)


        self.log_exposure_map = log_exposure_vals

        # Have to interpolate exposure not log_exposure due to possible -inf values
        self._exp_interpolator = RegularGridInterpolator(self.binning_geometry.axes, np.exp(self.log_exposure_map))
        
        return self.log_exposure_map
    
    def exp_interpolator(self, energy, lon, lat, *args, **kwargs):
        return self._exp_interpolator((energy, lon , lat), *args, **kwargs)

        

        
        

