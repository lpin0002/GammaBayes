import numpy as np
from astropy.io import fits
from .binning_geometry import GammaBinning
from gammapy.irf import load_irf_dict_from_file
import pickle
import pkg_resources

from .exposure import GammaLogExposure
from icecream import ic
from astropy import units as u

class GammaObs:
    def __init__(self, 
                 binning_geometry: GammaBinning, 
                 energy=[], lon=[], lat=[], 
                 pointing_dir:np.ndarray = None, 
                 observation_time: u.Quantity=None,
                 irf_loglike = None,
                 log_exposure: GammaLogExposure = None,
                 meta:dict = {}, **kwargs):
        
        self.energy = energy
        self.lon = lon
        self.lat = lat
        self.log_exposure = log_exposure
        self.binning_geometry = binning_geometry
        if len(energy):
            self.binned_data, _ = self._bin_data()
        else:
            self.binned_data = np.zeros(shape=(*self.binning_geometry.axes_dim,))





        self.meta = meta

        self.meta.update(kwargs)

        self.observation_time = observation_time


        if self.observation_time is None:
            self.observation_time =  self.meta.get('observation_time', None)


        if not(pointing_dir is None):
            self.pointing_dir = pointing_dir
        
        elif pointing_dir is None:
            self.pointing_dir = self.meta.get('pointing_dir')

        if self.pointing_dir is None:
            # If pointing direction not given, the centre of the spatial axes is presumed and
            self.pointing_dir = np.array([
                np.mean(self.binning_geometry.lon_axis.value), 
                np.mean(self.binning_geometry.lat_axis.to(self.binning_geometry.lon_axis.unit).value)
                ])*self.binning_geometry.lon_axis.unit
        
        if not(irf_loglike is None):
            self.irf_loglike = irf_loglike
            
        elif ('irf_loglike' in self.meta):
            self.irf_loglike = self.meta['irf_loglike']

        # If any one of these is not within the meta-data given that the irf_loglike is not contained, 
            # then it will return a False ==> a 0, so the product will be 0. Must either have all or 
            # none will be used as all are necessary.
        elif np.prod([irf_str in self.meta for irf_str in ['edisp', 'psf', 'ccr_bkg', 'aeff']]):
            self.irf_loglike = {self.meta[irf_str] for irf_str in ['edisp', 'psf', 'ccr_bkg', 'aeff']}
        else:
            self.irf_loglike = {}

        if hasattr(self.irf_loglike, 'pointing_dir'):
            self.irf_loglike.pointing_dir = pointing_dir


        if 'hemisphere' in self.meta:
            self.hemisphere = self.meta['hemisphere']

        else:
            self.hemisphere = None


    @property
    def energy_axis(self):
        return self.binning_geometry.energy_axis
    
    @property
    def lon_axis(self):
        return self.binning_geometry.lon_axis
    
    @property
    def lat_axis(self):
        return self.binning_geometry.lat_axis

    @property
    def spatial_axes(self):
        return self.lon_axis, self.lat_axis

    @property
    def axes(self):
        return self.energy_axis, self.lon_axis, self.lat_axis


    def _bin_data(self):
        binned_data, edges = np.histogramdd([self.energy, self.lon, self.lat], bins=[self.binning_geometry.energy_edges, self.binning_geometry.lon_edges, self.binning_geometry.lat_edges])

        return binned_data, edges

    def get_binned_data(self):
        return self.binned_data
    

    @property
    def binned_energy(self):
        return np.sum(self.binned_data, axis=(1, 2)), self.binning_geometry.energy_axis


    @property
    def binned_spatial(self):
        return np.sum(self.binned_data, axis=0), (self.binning_geometry.lon_axis, self.binning_geometry.lat_axis)

    @property
    def binned_longitude(self):
        return np.sum(self.binned_data, axis=(0, 2)), self.binning_geometry.lon_axis


    @property
    def binned_latitude(self):
        return np.sum(self.binned_data, axis=(0, 1)), self.binning_geometry.lat_axis
    


    def peek(self, axs=None, count_scaling='linear', cmap='afmhot', hist1dcolor='tab:orange', grid=True, grid_kwargs={}, figsize=(12,6), **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm        
        import numpy as np



        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=figsize, **kwargs)
        

        axs[0].hist(self.energy_axis, bins=self.binning_geometry.energy_edges, weights=self.binned_energy[0], color=hist1dcolor, log=True)
        axs[0].set_xlabel(r'Energy ['+ self.binning_geometry.energy_axis.unit.to_string('latex_inline')+']')
        axs[0].set_ylabel('Counts')
        axs[0].set_xlim(np.array([self.binning_geometry.energy_edges[0].value, self.binning_geometry.energy_edges[-1].value]))
        axs[0].set_yscale(count_scaling)
        axs[0].set_xscale('log')


        if grid:
            full_grid_kwargs = {'which':'major', 'color':'grey', 'ls':'--', 'alpha':0.2, 'zorder':-100}
            full_grid_kwargs.update(grid_kwargs)
            
            axs[0].grid(**full_grid_kwargs)
        
        im = axs[1].imshow(self.binned_spatial[0].T, origin='lower', aspect='auto', extent=[
            self.binning_geometry.lon_edges[0].value, 
            self.binning_geometry.lon_edges[-1].value,
            self.binning_geometry.lat_edges[0].value,
            self.binning_geometry.lat_edges[-1].value,
            ], 
            norm=count_scaling,
            cmap=cmap)
        plt.colorbar(im, ax=axs[1], label='Counts')


        axs[1].set_xlabel(r'Longitude ['+self.binning_geometry.lon_axis.unit.to_string('latex_inline')+']')
        axs[1].set_ylabel(r'Latitude ['+self.binning_geometry.lat_axis.unit.to_string('latex_inline')+']')
        
        # Apply tight_layout to the figure associated with the axs
        axs[0].figure.tight_layout()


        return axs
    

    @property
    def nonzero_bin_indices(self):
        return np.where(self.binned_data>0)

    @property
    def nonzero_bin_data(self):
        nonzero_indices = self.nonzero_bin_indices

        # Coordinates, Counts
        return np.asarray([self.energy_axis[nonzero_indices[0]], self.lon_axis[nonzero_indices[1]], self.lat_axis[nonzero_indices[2]]]).T, self.binned_data[nonzero_indices]



    @property
    def nonzero_coordinate_data(self):
        nonzero_indices = self.nonzero_bin_indices

        # Coordinates, Counts
        return np.asarray([self.energy_axis[nonzero_indices[0]], self.lon_axis[nonzero_indices[1]], self.lat_axis[nonzero_indices[2]]]).T




    def __add__(self, other):
        """
        Enables the addition of two EventData objects using the '+' operator, concatenating their data arrays.

        Args:
            other (EventData): Another EventData object to be added to this one.

        Returns:
            EventData: A new EventData instance containing the combined data of both operands.

        Raises:
            NotImplemented: If 'other' is not an instance of EventData.
        """
        if not isinstance(other, GammaObs):
            return NotImplemented
        
        if not np.array_equal(self.pointing_dir, other.pointing_dir):
            raise ValueError("""When combining multiple sets of data into a single observation, 
it is assumed the pointing direction of said observation is the same.""")
        if self.binning_geometry!=other.binning_geometry:
            raise ValueError("""When combining multiple sets of data into a single observation, 
it is assumed that the binning geometries are the same.""")


        # Concatenate the attributes of the two instances
        new_energy = np.concatenate([self.energy, other.energy])
        new_lon = np.concatenate([self.lon, other.lon])
        new_lat = np.concatenate([self.lat, other.lat])


        # You might need to decide how to handle other attributes like obs_id, zenith_angle, etc.
        # For this example, I'm just taking them from the first instance.
        return GammaObs(energy=new_energy, lon=new_lon, lat=new_lat, 
                        pointing_dir=self.pointing_dir, 
                        irf_loglike=self.irf_loglike,
                        binning_geometry=self.binning_geometry
)
            
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
        
    def __len__(self):
        """
        Enables the use of the len() function on an EventData instance to obtain the number of events.

        Returns:
            int: The total number of events in the dataset, equivalent to Nevents.
        """
        return len(self.nonzero_bin_data[0])
    

    @property
    def Nevents(self):
        """
        Calculates the number of events by counting the elements in the energy array.

        Returns:
            int: The total number of events in the dataset.
        """
        return len(self.energy)
    


    # Support for indexing like a list or array
    def __getitem__(self, key: int | slice):
        """
        Enables indexing into the GammaOsbs instance like a list or array, allowing direct access to 
        the data array's elements.

        Args:
            key (int, slice): The index or slice of the data array to retrieve.

        Returns:
            ndarray: The data at the specified index or slice.
        """
        bin_data = self.nonzero_bin_data
        return bin_data[0][key], bin_data[1][key]
    


    # Iterator support to enable looping over the dataset
    def __iter__(self):
        """
        Resets the internal counter and returns the iterator object (self) to enable looping over the dataset.

        Returns:
            self: The instance itself to be used as an iterator.
        """
        self._current_datum_idx = 0  # Reset the index each time iter is called
        return self

    # Next method for iterator protocol
    def __next__(self):
        """
        Advances to the next item in the dataset. If the end of the dataset is reached, it raises a StopIteration.

        Raises:
            StopIteration: Indicates that there are no further items to return.

        Returns:
            ndarray: The next data item in the sequence.
        """
        if self._current_datum_idx < len(self):
            current_data = self[self._current_datum_idx][0]
            self._current_datum_idx += 1
            return current_data
        else:
            raise StopIteration
        
    def to_dict(self):
        return {
            'binning_geometry': self.binning_geometry.to_dict(),
            'binned_data': self.binned_data,
            'meta': self.meta,
            'pointing_dir': self.pointing_dir,
            'irf_loglike': self.irf_loglike
        }


    def save(self, filename):
        data_to_save = self.to_dict()
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        # Create an empty instance
        instance = cls(binning_geometry=GammaBinning.from_dict(data['binning_geometry']), 
                    energy=[], lon=[], lat=[],  # Empty raw samples
                    meta=data['meta'],
                    pointing_dir=data['pointing_dir'],
                    irf_loglike=data['irf_loglike'])
        
        # Load the binned data
        instance.binned_data = data['binned_data']
        
        # Recreate the samples from the binned data
        energy_samples, lon_samples, lat_samples = instance._recreate_samples_from_binned_data()
        instance.energy = energy_samples
        instance.lon = lon_samples
        instance.lat = lat_samples
        
        return instance


    def _recreate_samples_from_binned_data(self):
        # Extract bin centers
        energy_centers = self.binning_geometry.energy_axis.value
        lon_centers = self.binning_geometry.lon_axis.value
        lat_centers = self.binning_geometry.lat_axis.value
        
        # Prepare lists to hold the recreated samples
        energy_samples = []
        lon_samples = []
        lat_samples = []
        
        # Loop over the binned data to recreate the samples
        for i in range(len(energy_centers)):
            for j in range(len(lon_centers)):
                for k in range(len(lat_centers)):
                    count = self.binned_data[i, j, k]
                    if count > 0:
                        # Repeat the bin centers according to the bin count
                        energy_samples.extend([energy_centers[i]] * int(count))
                        lon_samples.extend([lon_centers[j]] * int(count))
                        lat_samples.extend([lat_centers[k]] * int(count))
        
        # Convert to numpy arrays and return
        return np.array(energy_samples) * self.binning_geometry.energy_axis.unit, \
            np.array(lon_samples) * self.binning_geometry.lon_axis.unit, \
            np.array(lat_samples) * self.binning_geometry.lat_axis.unit



class GammaObsCube:
    def __init__(self,
                 binning_geometry: GammaBinning, 
                 observations: list[GammaObs],  
                 pointing_dirs=None,
                 observation_times=None,
                 name:str='NoName', 
                 meta:dict = {},
                 **kwargs):
        
        self.name = name
        self.binning_geometry = binning_geometry
        self.observations = observations
        self.meta = meta
        self.pointing_dirs = pointing_dirs
        self.observation_times = observation_times

        self.meta.update(kwargs)


    def add_observation(self, gamma_obs: GammaObs):
        if not np.array_equal(gamma_obs.binning_geometry.energy_edges, self.binning_geometry.energy_edges) or \
           not np.array_equal(gamma_obs.binning_geometry.lon_edges, self.binning_geometry.lon_edges) or \
           not np.array_equal(gamma_obs.binning_geometry.lat_edges, self.binning_geometry.lat_edges):
            raise ValueError("The binning of the observation does not match the binning of the GammaCube.")
        
        self.observations.append(gamma_obs)



    def get_data(self):
        combined_data = sum(obs.get_binned_data() for obs in self.observations)
        return combined_data
    

    @property
    def log_exposures(self):
        return [obs.log_exposure for obs in self.observations]
    
    @property
    def irf_loglikes(self):
        return [obs.irf_loglike for obs in self.observations]

    @property
    def energy_axis(self):
        return self.binning_geometry.energy_axis
    
    @property
    def lon_axis(self):
        return self.binning_geometry.lon_axis
    
    @property
    def lat_axis(self):
        return self.binning_geometry.lat_axis

    @property
    def spatial_axes(self):
        return self.lon_axis, self.lat_axis

    @property
    def axes(self):
        return self.energy_axis, self.lon_axis, self.lat_axis

    @property
    def collapsed_energy(self):
        return np.sum(self.get_data(), axis=(1, 2)), self.binning_geometry.energy_axis


    @property
    def collapsed_spatial(self):
        return np.sum(self.get_data(), axis=(0)), (self.binning_geometry.lon_axis, self.binning_geometry.lat_axis)


    @property
    def collapsed_longitude(self):
        return np.sum(self.get_data(), axis=(0, 2)), self.binning_geometry.lon_axis



    @property
    def collapsed_latitude(self):
        return np.sum(self.get_data(), axis=(0, 1)), self.binning_geometry.lat_axis

    @property
    def collapsed_data(self):
        return self.get_data(), self.binning_geometry.axes
    

    @property
    def lon_pointing_dirs(self):
        return [pointing[0] for pointing in self.pointing_dirs]
    

    @property
    def lat_pointing_dirs(self):
        return [pointing[1] for pointing in self.pointing_dirs]
    

    @property
    def central_pointing(self):

        # Calculate the mean point (center of mass)
        mean_point = np.mean(self.pointing_dirs, axis=0)

        # Calculate the Euclidean distance from each coordinate to the mean point
        distances = np.linalg.norm(self.pointing_dirs - mean_point, axis=1)

        # Find the index of the coordinate with the minimum distance
        most_central_index = np.argmin(distances)

        # Get the most central coordinate
        most_central_coordinate = self.pointing_dirs[most_central_index]

        return most_central_coordinate






    def peek(self, count_scaling='linear', grid=True, grid_kwargs={'which':'major', 'color':'grey', 'ls':'--', 'alpha':0.2, 'zorder':-100}, cmap='afmhot',hist1dcolor ='tab:orange', **kwargs):
        
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm        
        from matplotlib.pyplot import get_cmap

        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), **kwargs)
        
        axs[0].hist(self.energy_axis, bins=self.binning_geometry.energy_edges, weights=self.collapsed_energy[0], log=True, color=hist1dcolor)
        axs[0].set_xlabel(r'Energy ['+ self.binning_geometry.energy_axis.unit.to_string('latex_inline')+']')
        axs[0].set_ylabel('Counts')
        axs[0].set_xlim(np.array([self.binning_geometry.energy_edges[0].value, self.binning_geometry.energy_edges[-1].value]))
        axs[0].set_xscale('log')
        axs[0].set_yscale(count_scaling)

        if grid:            
            axs[0].grid(**grid_kwargs)

        im = axs[1].imshow(self.collapsed_spatial[0].T, origin='lower', aspect='auto', extent=[
            self.binning_geometry.lon_edges[0].value, 
            self.binning_geometry.lon_edges[-1].value,
            self.binning_geometry.lat_edges[0].value,
            self.binning_geometry.lat_edges[-1].value,
            ], 
            cmap=cmap,
            norm=count_scaling)
        


        plt.colorbar(im, ax=axs[1], label='Counts')
        axs[1].set_xlabel(r'Longitude ['+self.binning_geometry.lon_axis.unit.to_string('latex_inline')+']')
        axs[1].set_ylabel(r'Latitude ['+self.binning_geometry.lat_axis.unit.to_string('latex_inline')+']')
        axs[1].invert_xaxis()
        
        fig.tight_layout()

        return fig, axs



    def copy(self):
        import copy
        return copy.deepcopy(self)
    

    def __repr__(self):
        return f"GammaCube(name={self.name}, \nbinning={self.binning_geometry}, \nNum observations={len(self.observations)}, \nmeta={self.meta})"


    
    # from gammapy.utils.fits import HDULocation, LazyFitsData
    # import astropy.units as u
    # from astropy.io import fits
    # from astropy.table import Table
    # from gammapy.maps import LabelMapAxis, Map, MapAxis, WcsGeom
    # from gammapy.modeling.models import DatasetModels, FoVBackgroundModel, Models
    # Cash statistic?
    

    def slice_by_energy(self, energy_min=None, energy_max=None):
        new_obs = []
        for obs in self.observations:
            mask = np.ones_like(obs.energy, dtype=bool)
            if energy_min is not None:
                mask &= (obs.energy >= energy_min)
            if energy_max is not None:
                mask &= (obs.energy <= energy_max)
            if np.any(mask):
                new_obs.append(GammaObs(obs.energy[mask], obs.lon[mask], obs.lat[mask], obs.binning_geometry))
        return GammaObsCube(name=self.name, binning=self.binning_geometry, observations=new_obs, meta=self.meta)

    @classmethod
    def from_fits(cls, file_name: str):
        # Placeholder for actual implementation
        raise NotImplemented()

    @classmethod
    def from_dir(cls, dir_name: str):
        # Placeholder for actual implementation
        raise NotImplemented()


    def to_dict(self):
        return {
            'name': self.name,
            'binning_geometry': self.binning_geometry.to_dict(),
            'meta': self.meta,
            'observations': [obs.to_dict() for obs in self.observations]
        }



    def save(self, filename):
        data_to_save = self.to_dict()
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)


    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        binning = GammaBinning.from_dict(data['binning_geometry'])
        name = data['name']
        meta = data['meta']
        
        observations = []
        for obs_dict in data['observations']:
            obs = GammaObs(
                binning=GammaBinning.from_dict(obs_dict['binning_geometry']),
                meta=obs_dict['meta'],
                pointing_dir=obs_dict['pointing_dir'],
                irf_loglike=obs_dict['irf_loglike']
            )
            obs.binned_data = obs_dict['binned_data']
            
            # Recreate the samples from the binned data
            energy_samples, lon_samples, lat_samples = obs._recreate_samples_from_binned_data()
            obs.energy = energy_samples
            obs.lon = lon_samples
            obs.lat = lat_samples
            
            observations.append(obs)
        
        return cls(binning=binning, observations=observations, name=name, meta=meta)
    

        # Iterator support to enable looping over the dataset
    def __iter__(self):
        """
        Resets the internal counter and returns the iterator object (self) to enable looping over the dataset.

        Returns:
            self: The instance itself to be used as an iterator.
        """
        self._current_datum_idx = 0  # Reset the index each time iter is called
        return self

    # Next method for iterator protocol
    def __next__(self):
        """
        Advances to the next item in the dataset. If the end of the dataset is reached, it raises a StopIteration.

        Raises:
            StopIteration: Indicates that there are no further items to return.

        Returns:
            ndarray: The next data item in the sequence.
        """
        if self._current_datum_idx < len(self):
            current_obs = self.observations[self._current_datum_idx]
            self._current_datum_idx += 1
            return current_obs
        else:
            raise StopIteration
        

    # Support for indexing like a list or array
    def __getitem__(self, key: int | slice):
        """
        Enables indexing into the GammaObsCube instance like a list or array, 
        allowing access to individual observations.

        Args:
            key (int, slice): The index or slice of the data array to retrieve.

        Returns:
            ndarray: The observation at the specified index or slice.
        """
        return self.observations[key]
    

    def __len__(self):
        """
        Enables the use of the len() function on an EventData instance to obtain the number of events.

        Returns:
            int: The total number of events in the dataset, equivalent to Nevents.
        """
        return len(self.observations)
    
    







