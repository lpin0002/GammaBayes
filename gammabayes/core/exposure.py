# Will contain a class for handling effective area, 
#   observation time and multi-observation bin overlaps
#   i.e. Source Flux Exposures

import numpy as np
from .binning_geometry import GammaBinning
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator


class GammaLogExposure:
    def __init__(self, 
                 binning_geometry:GammaBinning, 
                 irfs=None, 
                 log_exposure_map: np.ndarray = None,
                 pointing_dir: np.ndarray = None,
                 observation_times: float | np.ndarray=1., 
                 observation_times_unit: u.Unit = u.s,
                 ):


        self.irfs = irfs
        self.pointing_dir = pointing_dir

        if not(self.pointing_dir is None):
            self.irfs.pointing_dir = self.pointing_dir


        if self.irfs is not None:
            self.log_aeff = self.irfs.log_aeff
            self.aeff_units = self.irfs.aeff_units
        else:
            self.log_aeff = None
            self.aeff_units = None

        self.binning_geometry = binning_geometry
        
        if hasattr(self.observation_times, 'unit'):
            self.time_unit = self.observation_times.unit
        else:
            self.time_unit = observation_times_unit



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

        if self.binning_geometry!=other.binning_geometry:
            raise NotImplemented("Adding exposures for different binning geometries is currently not supported.")

        new_exposure_map = np.logaddexp(self.log_exposure_map, other.log_exposure_map)

        return GammaLogExposure(binning_geometry=self.binning_geometry, log_exposure_map=new_exposure_map)
        



    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
        

    def refresh(self):

        log_exposure_vals = self.log_aeff(*self.binning_geometry.axes_mesh, 
                            pointing_dir=self.pointing_dir)
        
        log_exposure_vals+=np.log(self.observation_times.to(self.time_unit))


        self.log_exposure_map = log_exposure_vals


        self.exp_interpolator = RegularGridInterpolator(self.binning_geometry.axes, np.exp(self.log_exposure_map))

        
        return self.log_exposure_map

        

        
        

