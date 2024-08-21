# Will contain a class for handling effective area, 
#   observation time and multi-observation bin overlaps
#   i.e. Source Flux Exposures

import numpy as np
from .binning_geometry import GammaBinning
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator
import copy
from icecream import ic

def trivial_log_aeff(energy, lon, lat, pointing_dir=None):
    return energy.value*0.

class GammaLogExposure:
    def __init__(self, 
                 binning_geometry:GammaBinning, 
                 irfs=None, 
                 log_exposure_map: np.ndarray = None,
                 pointing_dir: np.ndarray = None,
                 observation_time: float | np.ndarray=None, 
                 observation_time_unit: u.Unit = None,
                 use_log_aeff: bool = True
                 ):
        
        # I commonly use None for default arguments and then follow up with this
            # so that one doesn't have to remember default arguments to allow them
            # to be used in other functions/classes
        self.use_log_aeff = use_log_aeff

        if isinstance(log_exposure_map, GammaLogExposure):
            self = copy.deepcopy(log_exposure_map)
        else:
            self.binning_geometry = binning_geometry

            if observation_time is None:
                observation_time = 1.*u.hr
            if observation_time_unit is None:
                observation_time_unit = 1*u.hr

            self.irfs = irfs
            self.pointing_dir = pointing_dir

            if not(self.pointing_dir is None) and (self.irfs is not None):
                self.irfs.pointing_dir = self.pointing_dir 
            elif not(self.irfs is None) and (self.pointing_dir is None):
                self.pointing_dir = self.irfs.pointing_dir

            else:
                self.pointing_dir = self.binning_geometry.spatial_centre



            if self.irfs is not None:
                self.log_aeff = self.irfs.log_aeff
                self.aeff_units = self.irfs.aeff_units
            else:
                self.log_aeff = trivial_log_aeff
                self.aeff_units = u.Unit("")

            
            if hasattr(observation_time, 'unit'):
                self.time_unit = observation_time.unit
            else:
                self.time_unit = observation_time_unit


            self.unit = self.time_unit*self.aeff_units



            self.observation_time = observation_time


            self._reset_cache()

            if not(log_exposure_map is None):
                self.log_exposure_map = log_exposure_map
                self._exp_interpolator = RegularGridInterpolator(self.binning_geometry.axes, np.exp(self.log_exposure_map))
            else:
                self.refresh()


    def __call__(self, *args, **kwargs):
        return np.log(self.exp_interpolator(*args, **kwargs).value)


    # Support for indexing like a list or array
    def __getitem__(self, key: int | slice):

        return self.log_exposure_map[key]

    def __add__(self, other):

        # if isinstance(other, GammaLogExposure):

        #     if self.binning_geometry!=other.binning_geometry:
        #         raise NotImplemented("Adding exposures for different binning geometries is currently not supported.")


        #     other_unit_scaling = (self.aeff_units/other.aeff_units).to("")

        #     new_exposure_map = np.logaddexp(self.log_exposure_map, np.log(other_unit_scaling)+other.log_exposure_map)


        #     return GammaLogExposure(binning_geometry=self.binning_geometry, 
        #                             log_exposure_map=new_exposure_map,
        #                             irfs=self.irfs,
        #                             pointing_dir=np.mean([self.pointing_dir, other.pointing_dir], axis=0),
        #                             observation_time=self.observation_time+other.observation_time,
        #                             )
        # else:

        if not(self._same_as_cached()):
            self.refresh()

        return other+self.log_exposure_map
        


    def _same_as_cached(self, pointing_dir=None, observation_time=None):

        if pointing_dir is None:
            pointing_dir = self.pointing_dir
        if observation_time is None:
            observation_time = self.observation_time

        same_as_cached = True

        same_as_cached = same_as_cached and np.array_equiv(pointing_dir, self._cached_pointing_dir)
        same_as_cached = same_as_cached and np.equal(observation_time, self._cached_observation_time)

        return same_as_cached
    

    def _reset_cache(self):


        if self.pointing_dir is None:
            raise ValueError()

        self._cached_pointing_dir = self.pointing_dir
        self._cached_observation_time = self.observation_time



    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
        

    def refresh(self):

        if self.use_log_aeff:
            log_exposure_vals = self.log_aeff(*self.binning_geometry.axes_mesh, pointing_dir=self.pointing_dir)
        else:
            log_exposure_vals = self.binning_geometry.axes_mesh[0].value*0
        
        log_exposure_vals+=np.log(self.observation_time.to(self.time_unit).value)


        self.log_exposure_map = log_exposure_vals

        # Have to interpolate exposure not log_exposure due to possible -inf values
        self._exp_interpolator = RegularGridInterpolator(self.binning_geometry.axes, np.exp(self.log_exposure_map))

        self._reset_cache()
        
        return self.log_exposure_map
    
    def exp_interpolator(self, energy, lon, lat, *args, pointing_dir=None, observation_time=None, **kwargs):

        if self._same_as_cached(pointing_dir, observation_time):
            return self._exp_interpolator((energy, lon , lat), *args, **kwargs)*self.unit
        else:
            return np.exp(self.log_aeff(energy, lon, lat, pointing_dir=pointing_dir)+np.log(observation_time.to(self.time_unit).value))*self.unit
        

    def peek(self, fig_kwargs={}, pcolormesh_kwargs={}, **kwargs):

        from matplotlib import pyplot as plt
        from matplotlib import patches
        from matplotlib.colors import LogNorm
        from gammabayes.utils.integration import iterate_logspace_integration

        fig_kwargs.update(kwargs)
        integrated_exposure = iterate_logspace_integration(logy = self.log_exposure_map, 
                                                        axes=(self.binning_geometry.energy_axis.value,), 
                                                        axisindices=[0])
        
        if 'norm' not in pcolormesh_kwargs:
            pcolormesh_kwargs['norm'] = 'log'

        plt.figure(**fig_kwargs)
        plt.pcolormesh(self.binning_geometry.lon_axis.value, self.binning_geometry.lat_axis.value, integrated_exposure, **pcolormesh_kwargs)
        plt.colorbar(label=r"Exposure ["+self.unit.to_string('latex')+"]")
        plt.xlabel(r"Longitude ["+self.binning_geometry.lon_axis.unit.to_string('latex')+']')
        plt.ylabel(r"Latitude ["+self.binning_geometry.lat_axis.unit.to_string('latex')+']')

        plt.show()
            

        

        
        

