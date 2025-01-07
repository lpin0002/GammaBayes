import numpy as np
from .binning_geometry import GammaBinning
import pickle
from .exposure import GammaLogExposure
from astropy import units as u
from numpy.typing import ArrayLike
import time
from collections import defaultdict
from icecream import ic
from .gamma_obs import GammaObs

class GammaObsCube:
    """
    A class representing a collection of gamma-ray observations, their associated binning geometry, 
    and metadata. Provides functionality to manage, analyze, and visualize the combined data 
    from multiple observations.

    Attributes:
        name (str): The name of the cube.
        binning_geometry (GammaBinning): The binning geometry for spatial and energy bins.
        observations (list[GammaObs]): A list of GammaObs instances.
        meta (dict): Metadata associated with the observation cube.
        pointing_dirs (list): List of pointing directions for each observation.
        live_times (list): List of live times for each observation.
    """

    def __init__(self,
                 observations: list[GammaObs],  
                 binning_geometry: GammaBinning = None, 
                 pointing_dirs = None,
                 live_times = None,
                 name: str = 'NoName', 
                 meta: dict = {},
                 **kwargs):
        """
        Initialize the GammaObsCube with observations, optional binning geometry, and metadata.

        Args:
            observations (list[GammaObs]): List of GammaObs instances.
            binning_geometry (GammaBinning, optional): Shared binning geometry. Defaults to the geometry of the first observation.
            pointing_dirs (list, optional): Pointing directions for the observations.
            live_times (list, optional): Live times for the observations.
            name (str, optional): Name of the cube. Defaults to 'NoName'.
            meta (dict, optional): Metadata dictionary. Defaults to an empty dictionary.
            **kwargs: Additional metadata to be included.
        """
        self.name = name
        self.binning_geometry = binning_geometry or observations[0].binning_geometry
        self.observations = observations
        self.meta = meta

        self.pointing_dirs = pointing_dirs or [obs.pointing_dirs for obs in observations]
        self.live_times = live_times or [obs.live_times for obs in observations]
        self.meta.update(kwargs)

    def add_observation(self, gamma_obs: GammaObs):
        """
        Add a GammaObs observation to the cube. Ensures the binning geometry matches.

        Args:
            gamma_obs (GammaObs): The observation to be added.

        Raises:
            ValueError: If the binning geometry does not match.
        """
        if gamma_obs.binning_geometry != self.binning_geometry:
            raise ValueError("The binning of the observation does not match the binning of the GammaCube.")
        self.observations.append(gamma_obs)

    def get_data(self):
        """
        Combine binned data from all observations.

        Returns:
            ndarray: Combined binned data array.
        """
        combined_data = sum(obs.get_binned_data() for obs in self.observations)
        return combined_data

    @property
    def log_exposures(self):
        """
        Retrieve log exposures for all observations.

        Returns:
            list[GammaLogExposure]: List of log exposures.
        """
        return [obs.log_exposure for obs in self.observations]

    def _calc_combined_log_exposure(self):
        """
        Calculate and cache the combined log exposure for all observations.
        """
        combined_log_exposure = sum(self.log_exposures)
        combined_log_exposure.pointing_dirs = self.binning_geometry.spatial_centre
        self._combined_log_exposure = combined_log_exposure

    @property
    def combined_log_exposure(self):
        """
        Retrieve the combined log exposure, calculating it if necessary.

        Returns:
            GammaLogExposure: The combined log exposure.
        """
        if not hasattr(self, "_combined_log_exposure"):
            self._calc_combined_log_exposure()
        return self._combined_log_exposure

    @property
    def irf_loglikes(self):
        """
        Retrieve IRF log-likelihoods for all observations.

        Returns:
            list: List of IRF log-likelihoods.
        """
        return [obs.irf_loglike for obs in self.observations]

    @property
    def collapsed_energy(self):
        """
        Sum data over spatial axes, collapsing along the energy axis.

        Returns:
            tuple: Summed data and energy axis.
        """
        return np.sum(self.get_data(), axis=(1, 2)), self.binning_geometry.energy_axis

    @property
    def collapsed_spatial(self):
        """
        Sum data over the energy axis, collapsing along spatial axes.

        Returns:
            tuple: Summed data and spatial axes.
        """
        return np.sum(self.get_data(), axis=0), (self.binning_geometry.lon_axis, self.binning_geometry.lat_axis)

    @property
    def collapsed_lon(self):
        """
        Sum data over latitude and energy axes, collapsing along longitude.

        Returns:
            tuple: Summed data and longitude axis.
        """
        return np.sum(self.get_data(), axis=(0, 2)), self.binning_geometry.lon_axis

    @property
    def collapsed_lat(self):
        """
        Sum data over longitude and energy axes, collapsing along latitude.

        Returns:
            tuple: Summed data and latitude axis.
        """
        return np.sum(self.get_data(), axis=(0, 1)), self.binning_geometry.lat_axis

    @property
    def collapsed_data(self):
        """
        Retrieve the combined data and binning axes.

        Returns:
            tuple: Combined data and binning axes.
        """
        return self.get_data(), self.binning_geometry.axes

    @property
    def lon_pointing_dirs(self):
        """
        Retrieve the longitude pointing directions.

        Returns:
            list: Longitude pointing directions.
        """
        return [pointing[0] for pointing in self.pointing_dirs]

    @property
    def lat_pointing_dirs(self):
        """
        Retrieve the latitude pointing directions.

        Returns:
            list: Latitude pointing directions.
        """
        return [pointing[1] for pointing in self.pointing_dirs]

    @property
    def central_pointing(self):
        """
        Calculate the most central pointing direction based on the mean of all pointing directions.

        Returns:
            ndarray: The most central pointing direction.
        """
        mean_point = np.mean(self.pointing_dirs, axis=0)
        distances = np.linalg.norm(self.pointing_dirs - mean_point, axis=1)
        most_central_index = np.argmin(distances)
        return self.pointing_dirs[most_central_index]

    def peek(self, count_scaling='linear', grid=True, grid_kwargs={'which': 'major', 'color': 'grey', 'ls': '--', 'alpha': 0.2, 'zorder': -100}, cmap='afmhot', hist1dcolor='tab:orange', **kwargs):
        """
        Generate a quick plot of the data, including a histogram and a spatial map.

        Args:
            count_scaling (str, optional): Scaling for counts. Defaults to 'linear'.
            grid (bool, optional): Whether to include grid lines. Defaults to True.
            grid_kwargs (dict, optional): Grid styling parameters.
            cmap (str, optional): Colormap for the spatial map. Defaults to 'afmhot'.
            hist1dcolor (str, optional): Color for the histogram. Defaults to 'tab:orange'.
            **kwargs: Additional matplotlib parameters for the plot.

        Returns:
            tuple: Figure and axis objects.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        fig, axs = plt.subplots(1, 2, **kwargs)
        log = count_scaling != 'linear'

        axs[0].hist(self.binning_geometry.energy_axis, bins=self.binning_geometry.energy_edges, weights=self.collapsed_energy[0], log=log, color=hist1dcolor)
        axs[0].set_xlabel(r'Energy [' + self.binning_geometry.energy_axis.unit.to_string('latex_inline') + ']')
        axs[0].set_ylabel('Counts')
        axs[0].set_xlim([self.binning_geometry.energy_edges[0].value, self.binning_geometry.energy_edges[-1].value])
        axs[0].set_xscale('log')
        axs[0].set_yscale(count_scaling)
        if grid:
            axs[0].grid(**grid_kwargs)

        im = axs[1].imshow(self.collapsed_spatial[0].T, origin='lower', aspect='auto', extent=[
            self.binning_geometry.lon_edges[0].value,
            self.binning_geometry.lon_edges[-1].value,
            self.binning_geometry.lat_edges[0].value,
            self.binning_geometry.lat_edges[-1].value,
        ], cmap=cmap, norm=LogNorm() if log else None)

        plt.colorbar(im, ax=axs[1], label='Counts')
        axs[1].set_xlabel(r'Longitude [' + self.binning_geometry.lon_axis.unit.to_string('latex_inline') + ']')
        axs[1].set_ylabel(r'Latitude [' + self.binning_geometry.lat_axis.unit.to_string('latex_inline') + ']')
        axs[1].invert_xaxis()
        fig.tight_layout()

        return fig, axs

    def copy(self):
        """
        Create a deep copy of the GammaObsCube instance.

        Returns:
            GammaObsCube: A deep copy of the current instance.
        """
        import copy
        return copy.deepcopy(self)

    def __repr__(self):
        """
        Generate a string representation of the GammaObsCube instance.

        Returns:
            str: String representation of the instance.
        """
        return f"GammaCube(name={self.name}, \nbinning={self.binning_geometry}, \nNum observations={len(self.observations)}, \nmeta={self.meta})"

    def slice_by_energy(self, energy_min=None, energy_max=None):
        """
        Create a new GammaObsCube with observations filtered by energy range.

        Args:
            energy_min (float, optional): Minimum energy threshold.
            energy_max (float, optional): Maximum energy threshold.

        Returns:
            GammaObsCube: A new instance with observations within the energy range.
        """
        new_obs = []
        for obs in self.observations:
            mask = np.ones_like(obs.energy, dtype=bool)
            if energy_min is not None:
                mask &= (obs.energy >= energy_min)
            if energy_max is not None:
                mask &= (obs.energy <= energy_max)
            if np.any(mask):
                new_obs.append(GammaObs(obs.energy[mask], obs.lon[mask], obs.lat[mask], obs.binning_geometry))
        return GammaObsCube(name=self.name, binning_geometry=self.binning_geometry, observations=new_obs, meta=self.meta)

    @classmethod
    def from_fits(cls, file_name: str):
        """
        Create a GammaObsCube instance from a FITS file.

        Args:
            file_name (str): Path to the FITS file.

        Raises:
            NotImplementedError: Placeholder for actual implementation.
        """
        raise NotImplementedError()

    @classmethod
    def from_fits_dir(cls, dir_name: str):
        """
        Create a GammaObsCube instance from a directory of FITS files.

        Args:
            dir_name (str): Path to the directory.

        Raises:
            NotImplementedError: Placeholder for actual implementation.
        """
        raise NotImplementedError()

    def to_dict(self, include_irfs=False, include_obs_meta=False):
        """
        Convert the GammaObsCube instance to a dictionary format.

        Args:
            include_irfs (bool, optional): Whether to include IRF data. Defaults to False.
            include_obs_meta (bool, optional): Whether to include observation metadata. Defaults to False.

        Returns:
            dict: Dictionary representation of the instance.
        """
        return {
            'name': self.name,
            'binning_geometry': self.binning_geometry.to_dict(),
            'meta': self.meta,
            'observations': [obs.to_dict(save_irf=include_irfs, include_meta=include_obs_meta) for obs in self.observations]
        }

    def save(self, filename, save_irfs=False, save_obs_meta=False):
        """
        Save the GammaObsCube instance to a file.

        Args:
            filename (str): Path to the file.
            save_irfs (bool, optional): Whether to include IRF data. Defaults to False.
            save_obs_meta (bool, optional): Whether to include observation metadata. Defaults to False.
        """
        data_to_save = self.to_dict(include_irfs=save_irfs, include_obs_meta=save_obs_meta)
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)

    @classmethod
    def load(cls, filename):
        """
        Load a GammaObsCube instance from a file.

        Args:
            filename (str): Path to the file.

        Returns:
            GammaObsCube: The loaded instance.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        binning = GammaBinning.from_dict(data['binning_geometry'])
        name = data['name']
        meta = data.get('meta', {})

        observations = []
        for obs_dict in data['observations']:
            obs_meta_dict = obs_dict.get('meta', {})
            obs_meta_dict.update(meta)

            obs = GammaObs(
                binning=binning,
                meta=obs_meta_dict,
                pointing_dir=obs_dict['pointing_dir'],
                irf_loglike=obs_dict.get('irf_loglike')
            )
            obs.binned_data = obs_dict['binned_data']

            energy_samples, lon_samples, lat_samples = obs._recreate_samples_from_binned_data()
            obs.energy = energy_samples
            obs.lon = lon_samples
            obs.lat = lat_samples

            observations.append(obs)

        return cls(binning_geometry=binning, observations=observations, name=name, meta=meta)

    def __iter__(self):
        """
        Reset the internal counter and return the iterator object to enable looping.

        Returns:
            GammaObsCube: The instance itself to be used as an iterator.
        """
        self._current_datum_idx = 0
        return self

    def __next__(self):
        """
        Advance to the next observation in the cube.

        Raises:
            StopIteration: When no more observations are available.

        Returns:
            GammaObs: The next observation in the cube.
        """
        if self._current_datum_idx < len(self):
            current_obs = self.observations[self._current_datum_idx]
            self._current_datum_idx += 1
            return current_obs
        else:
            raise StopIteration

    def __getitem__(self, key: int | slice):
        """
        Enable indexing into the GammaObsCube to access observations.

        Args:
            key (int, slice): Index or slice to retrieve observations.

        Returns:
            GammaObs | list[GammaObs]: The requested observation(s).
        """
        return self.observations[key]

    def __len__(self):
        """
        Get the number of observations in the cube.

        Returns:
            int: Number of observations.
        """
        return len(self.observations)
