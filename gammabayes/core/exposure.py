# Will contain a class for handling effective area, 
#   observation time and multi-observation bin overlaps
#   i.e. Source Flux Exposures

import numpy as np
from .binning_geometry import GammaBinning
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator
import copy
from warnings import warn



def trivial_log_aeff(energy, lon, lat, pointing_dir=None):
    return energy.value*0.

class GammaLogExposure:
    def __init__(self, 
                 binning_geometry:GammaBinning, 
                 irfs=None, 
                 log_exposure_map: np.ndarray = None,
                 pointing_dirs: list[np.ndarray[u.Quantity]] | np.ndarray[u.Quantity] = None,

                 # One second is taken as most differential fluxes are per second, so this works as a unit where the absolute values are the quantities don't change
                 live_times: u.Quantity | np.ndarray[u.Quantity]=1*u.s, 
                 use_log_aeff: bool = True,

                 # By default the units of the effective area are taken to be u.m**2 and intermediary calculations use seconds when needed
                 unit_bases = None, 
                 ):
        if unit_bases is None:
            unit_bases = [u.m, u.s] 

        np.seterr(divide='ignore')
        
        self.use_log_aeff = use_log_aeff

        if isinstance(log_exposure_map, GammaLogExposure):
            for attr, value in log_exposure_map.__dict__.items():
                setattr(self, attr, copy.deepcopy(value))
        else:
            self.binning_geometry = binning_geometry

            self.irfs = irfs


            if hasattr(self.irfs, "log_aeff"):
                self.log_aeff = self.irfs.log_aeff
                self.aeff_units = self.irfs.aeff_units
            else:
                self.log_aeff = trivial_log_aeff
                self.aeff_units = u.Unit("")



            self.__parse_livetime(live_times=live_times)
            self.__parse_pointing_dirs(pointing_dirs=pointing_dirs)

                
            self.unit = self.live_time_units*self.aeff_units


            # Gets rid of annoying things like u.hr/u.s not being simplified
            try:
                decomposed_unit = self.unit.decompose(unit_bases)
                self.unit = ((1*decomposed_unit).decompose(unit_bases)).unit
                self.log_unit_converter = np.log(((1*decomposed_unit).decompose(unit_bases)).value)
            except:
                self.unit = self.unit
                self.log_unit_converter = 0.
                


            self._reset_cache()

            if not(log_exposure_map is None):
                self.log_exposure_map = log_exposure_map
                self._exp_interpolator = RegularGridInterpolator(self.binning_geometry.axes, np.exp(self.log_exposure_map), 
                                                                 bounds_error=False, fill_value=0)
            else:
                self.refresh()

                

    def __parse_livetime(self, live_times):

        if live_times is None:
            live_times = 1.*u.Unit("")

        if isinstance(live_times, u.Quantity):
            if live_times.isscalar:
                live_times = [live_times]
            else:
                live_times = live_times
        elif isinstance(live_times, (int, float)):
            live_times = [live_times]
        elif isinstance(live_times, (np.ndarray, list, tuple)):
            live_times = live_times
        else:
            warn("Cannot interpret livetime input. Must be list of scalars or scalar. Not sure how you gave something else.")
        
        self.live_times = live_times


        try:
            self.live_time_units = self.live_times[0].unit
        except:
            self.live_time_units = u.Unit("")

        try:
            self.live_times_values = [live_time.to(self.live_time_units).value for live_time in self.live_times]
        except:
            self.live_times_values = [live_time for live_time in self.live_times]


        self.live_times = np.array(self.live_times_values)*self.live_time_units



    def __parse_pointing_dirs(self, pointing_dirs):

        if np.asarray(pointing_dirs).ndim <2:
            pointing_dirs = [pointing_dirs]

        self.pointing_dirs = pointing_dirs

        if hasattr(self.irfs, "pointing_dir") and (self.pointing_dirs is None):
            self.pointing_dirs = [self.irfs.pointing_dir]

        elif self.pointing_dirs is None:
            self.pointing_dirs = [self.binning_geometry.spatial_centre]


    def __call__(self, *args, **kwargs):
        return np.log(self.exp_interpolator(*args, **kwargs).value)+self.log_unit_converter


    # Support for indexing like a list or array
    def __getitem__(self, key: int | slice):

        return self.log_exposure_map[key]

    def __add__(self, other):

        if isinstance(other, GammaLogExposure):

            if self.binning_geometry!=other.binning_geometry:
                raise NotImplemented("Adding exposures for different binning geometries is currently not supported.")


            other_unit_scaling = (self.aeff_units/other.aeff_units).to("")

            new_exposure_map = np.logaddexp(self.log_exposure_map, np.log(other_unit_scaling)+other.log_exposure_map)


            return GammaLogExposure(binning_geometry=self.binning_geometry, 
                                    log_exposure_map=new_exposure_map,
                                    irfs=self.irfs,
                                    pointing_dirs= self.pointing_dirs.extend(other.pointing_dirs),
                                    lives_times=self.live_times.extend(other.live_times),
                                    )
        else:

            return other+self.log_exposure_map
        


    def _same_as_cached(self, pointing_dirs: np.ndarray[u.Quantity]| u.Quantity=None, live_times:u.Quantity=None):


        if pointing_dirs is None:
            same_as_cached = True
            return same_as_cached
        
        if np.asarray(pointing_dirs).ndim <2:

            same_as_cached = np.any((np.array(self.pointing_dirs) == pointing_dirs.value).all(axis=1))

            return same_as_cached

        same_as_cached = np.array_equiv(np.sort(pointing_dirs, axis=0), np.sort(self.pointing_dirs, axis=0))


        return same_as_cached
    

    def _reset_cache(self):


        if self.pointing_dirs is None:
            raise ValueError()

        self._cached_pointing_dirs = self.pointing_dirs
        self._cached_live_times = self.live_times



    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
        

    def refresh(self):

        if self.use_log_aeff:
            log_exposure_vals = -np.inf

            for pointing_dir, live_time in zip(self.pointing_dirs, self.live_times):
                log_exposure_vals = np.logaddexp(log_exposure_vals, self.log_aeff(*self.binning_geometry.axes_mesh, pointing_dir=pointing_dir)+np.log(live_time.value)+self.log_unit_converter)

        else:
            log_exposure_vals = self.binning_geometry.axes_mesh[0].value*0+np.log(live_time.value)+self.log_unit_converter
        

        self.log_exposure_map = log_exposure_vals

        # Have to interpolate exposure not log_exposure due to possible -inf values
        self._exp_interpolator = RegularGridInterpolator(self.binning_geometry.axes, np.exp(self.log_exposure_map), bounds_error=False, fill_value=0)

        self._reset_cache()
        
        return self.log_exposure_map
    

    def add_single_exposure(self, pointing_dir:np.ndarray[u.Quantity], live_time:u.Quantity):


        self.pointing_dirs = self.pointing_dirs.append(pointing_dir)
        self.live_times = self.live_times.append(live_time)

        if hasattr(live_time, "unit"):
            live_time = live_time.to(self.live_time_units)
        else:
            live_time = live_time*self.live_time_units


        if self.use_log_aeff:

            self.log_exposure_map = np.logaddexp(self.log_exposure_map, self.log_aeff(*self.binning_geometry.axes_mesh, pointing_dir=pointing_dir)+np.log(live_time.value)+self.log_unit_converter)

        else:
            self.log_exposure_map = np.logaddexp(self.log_exposure_map, self.binning_geometry.axes_mesh[0].value*0+np.log(live_time.value)+self.log_unit_converter)


        # Have to interpolate exposure not log_exposure due to possible -inf values
        self._exp_interpolator = RegularGridInterpolator(self.binning_geometry.axes, np.exp(self.log_exposure_map))

        self._reset_cache()
        
        return self.log_exposure_map
            
    
    def exp_interpolator(self, energy, lon, lat, *args, pointing_dirs=None, live_times=None, **kwargs):

        if not self._same_as_cached(pointing_dirs=pointing_dirs):
            if np.array(pointing_dirs).size>2:
                self.pointing_dirs = pointing_dirs
                self.live_times = live_times

                self.refresh()

            else:
                self.add_single_exposure(pointing_dir=pointing_dirs, live_times=live_times)

        return self._exp_interpolator((energy, lon , lat), *args, **kwargs)*self.unit
        

    def peek(self, fig_kwargs=None, pcolormesh_kwargs=None, plot_kwargs=None, **kwargs):

        if fig_kwargs is None:
            fig_kwargs = {}

        if pcolormesh_kwargs is None:
            pcolormesh_kwargs = {}

        if plot_kwargs is None:
            plot_kwargs = {}

        from matplotlib import pyplot as plt
        from matplotlib import patches
        from matplotlib.colors import LogNorm
        from gammabayes.utils.integration import iterate_logspace_integration

        fig_kwargs.update(kwargs)

        if 'figsize' not in fig_kwargs:
            fig_kwargs['figsize'] = (12, 6)
        integrated_energy_exposure = iterate_logspace_integration(logy = self.log_exposure_map, 
                                                        axes=(self.binning_geometry.energy_axis.value,), 
                                                        axisindices=[0])
        

        integrated_spatial_exposure = iterate_logspace_integration(logy = self.log_exposure_map, 
                                                axes=(self.binning_geometry.lon_axis.value, self.binning_geometry.lat_axis.value,), 
                                                axisindices=[1, 2])
        

        integrated_lat_exposure = iterate_logspace_integration(logy = self.log_exposure_map, 
                                                axes=(self.binning_geometry.lat_axis.value,), 
                                                axisindices=[2])

        weighted_mean_pointing_dir = np.sum(np.asarray(self.pointing_dirs).T*self.live_times, axis=1)/np.sum(self.live_times)

        try:
            weighted_mean_pointing_dir = [weighted_mean_pointing_dir[0].value, weighted_mean_pointing_dir[1].value]
        except:
            weighted_mean_pointing_dir = weighted_mean_pointing_dir

        energy_slice = np.abs(self.binning_geometry.energy_axis.value-1).argmin()
        lon_slice = np.abs(self.binning_geometry.lon_axis.value-weighted_mean_pointing_dir[0]).argmin()
        lat_slice = np.abs(self.binning_geometry.lat_axis.value-weighted_mean_pointing_dir[1]).argmin()

        energy_slice_val = self.binning_geometry.energy_axis.value[energy_slice]
        lon_slice_val = self.binning_geometry.lon_axis.value[lon_slice]
        lat_slice_val = self.binning_geometry.lon_axis.value[lat_slice]


        slice_coord_str = f"({lon_slice_val:.2g}, {lat_slice_val:.2g})"
        
        if 'norm' not in pcolormesh_kwargs:
            pcolormesh_kwargs['norm'] = 'log'

        fig, ax = plt.subplots(2, 3, **fig_kwargs)


        ax[0,0].plot(self.binning_geometry.energy_axis.value, np.exp(self.log_exposure_map[:, lon_slice, lat_slice].T), label=f'Exposure at {slice_coord_str}', **plot_kwargs)
        ax[0,0].set_xscale('log')
        ax[0,0].set_xlabel(r"Energy ["+self.binning_geometry.energy_axis.unit.to_string('latex')+']')
        ax[0,0].set_ylabel(r"Exposure ["+(self.unit).to_string('latex')+"]",)
        ax[0,0].legend()
        ax[0,0].grid(which='major', c='grey', ls='--', alpha=0.4)


        ax[1,0].plot(self.binning_geometry.energy_axis.value, np.exp(integrated_spatial_exposure.T), **plot_kwargs)
        ax[1,0].set_xscale('log')
        ax[1,0].set_xlabel(r"Energy ["+self.binning_geometry.energy_axis.unit.to_string('latex')+']')
        ax[1,0].set_ylabel(r"Integrated Exposure ["+(self.unit*self.binning_geometry.lon_axis.unit*self.binning_geometry.lon_axis.unit).to_string('latex')+"]",)

        pcm = ax[0,1].pcolormesh(self.binning_geometry.lon_axis.value, self.binning_geometry.lat_axis.value, 
                                 np.exp(self.log_exposure_map[energy_slice, :, :].T),
                                 **pcolormesh_kwargs)
        ax[0,1].legend(title="Slice at 1 TeV")
        plt.colorbar(mappable=pcm, label=r"Exposure ["+(self.unit).to_string('latex')+"]", ax= ax[0,1])
        ax[0,1].set_xlabel(r"Longitude ["+self.binning_geometry.lon_axis.unit.to_string('latex')+']')
        ax[0,1].set_ylabel(r"Latitude ["+self.binning_geometry.lat_axis.unit.to_string('latex')+']')
        ax[0,1].set_aspect('equal', adjustable='box')
        ax[0,1].invert_xaxis()


        int_pcm = ax[1,1].pcolormesh(self.binning_geometry.lon_axis.value, self.binning_geometry.lat_axis.value, np.exp(integrated_energy_exposure.T), **pcolormesh_kwargs)
        plt.colorbar(mappable=int_pcm, label=r"Integrated Exposure ["+(self.unit*self.binning_geometry.energy_axis.unit).to_string('latex')+"]", ax= ax[1,1])
        ax[1,1].set_xlabel(r"Longitude ["+self.binning_geometry.lon_axis.unit.to_string('latex')+']')
        ax[1,1].set_ylabel(r"Latitude ["+self.binning_geometry.lat_axis.unit.to_string('latex')+']')
        ax[1,1].set_aspect('equal', adjustable='box')
        ax[1,1].invert_xaxis()



        pcm = ax[0,2].pcolormesh(self.binning_geometry.lon_axis.value, self.binning_geometry.energy_axis.value, 
                                 np.exp(self.log_exposure_map[:, :, lat_slice]),
                                 **pcolormesh_kwargs)
        ax[0,2].legend(title=f"Slice at lat={lat_slice_val:.2g} deg")
        plt.colorbar(mappable=pcm, label=r"Exposure ["+(self.unit).to_string('latex')+"]", ax= ax[0,2])
        ax[0,2].set_xlabel(r"Longitude ["+self.binning_geometry.lon_axis.unit.to_string('latex')+']')
        ax[0,2].invert_xaxis()
        ax[0,2].set_ylabel(r"Energy ["+self.binning_geometry.energy_axis.unit.to_string('latex')+']')
        ax[0,2].set_yscale('log')

        int_pcm = ax[1,2].pcolormesh(self.binning_geometry.lon_axis.value, self.binning_geometry.energy_axis.value, np.exp(integrated_lat_exposure), **pcolormesh_kwargs)
        plt.colorbar(mappable=int_pcm, label=r"Integrated Exposure ["+(self.unit*self.binning_geometry.lat_axis.unit).to_string('latex')+"]", ax= ax[1,2])
        ax[1,2].set_xlabel(r"Longitude ["+self.binning_geometry.lon_axis.unit.to_string('latex')+']')
        ax[1,2].invert_xaxis()
        ax[1,2].set_yscale('log')
        ax[1,2].set_ylabel(r"Energy ["+self.binning_geometry.energy_axis.unit.to_string('latex')+']')

        plt.tight_layout()

        return fig, ax

    @property
    def log_obs_time_map(self):
        return np.log(self.observation_time.to(self.observation_time_unit).value)
            
