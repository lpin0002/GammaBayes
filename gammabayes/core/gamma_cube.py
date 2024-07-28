import numpy as np
from astropy.io import fits
from .binning_geometry import GammaBinning
from gammapy.irf import load_irf_dict_from_file
import pickle
import pkg_resources

from .exposure import GammaLogExposure



class GammaObs:
    def __init__(self, binning: GammaBinning, energy=[], lon=[], lat=[], 
                 pointing_dir:np.ndarray = None, 
                 irf_loglike = None,
                 logexposure: GammaLogExposure = None,
                 meta:dict = {}, **kwargs):
        
        self.energy = energy
        self.lon = lon
        self.lat = lat
        self.binning = binning
        if len(energy):
            self.binned_data, _ = self._bin_data()
        else:
            self.binned_data = np.zeros(shape=(*self.binning.axes_dim,))

        self.meta = meta
        self.irf_loglike = irf_loglike

        self.meta.update(kwargs)

        if not(pointing_dir is None):
            self.pointing_dir = pointing_dir
            self.irf_loglike.pointing_dir = pointing_dir


        elif (pointing_dir is None) and 'pointing_dir' in self.meta:
            self.pointing_dir = self.meta['pointing_dir']
        else:
            # If pointing direction not given, the centre of the spatial axes is presumed and
            self.pointing_dir = np.array([np.mean(self.binning.lon_axis.value), np.mean(self.binning.lat_axis.to(self.binning.lon_axis.unit).value)])*self.binning.lon_axis.unit
        
        if 'irf_loglike' in self.meta:
            self.irfs = self.meta['irf_loglike']

        # If any one of these is not within the meta-data given that the irf_loglike is not contained, 
            # then it will return a False ==> a 0, so the product will be 0. Must either have all or 
            # none will be used as all are necessary.
        elif np.prod([irf_str in self.meta for irf_str in ['edisp', 'psf', 'ccr_bkg', 'aeff']]):
            self.irfs = {self.meta[irf_str] for irf_str in ['edisp', 'psf', 'ccr_bkg', 'aeff']}
        else:
            self.irfs = {}


        if 'obs_time' in self.meta:
            self.obs_time = self.meta['obs_time']

        else:
            self.obs_time = np.nan


        if 'hemisphere' in self.meta:
            self.hemisphere = self.meta['hemisphere']

        else:
            self.hemisphere = None


    @property
    def energy_axis(self):
        return self.binning.energy_axis
    
    @property
    def lon_axis(self):
        return self.binning.lon_axis
    
    @property
    def lat_axis(self):
        return self.binning.lat_axis

    @property
    def spatial_axes(self):
        return self.lon_axis, self.lat_axis

    @property
    def axes(self):
        return self.energy_axis, self.lon_axis, self.lat_axis


    def _bin_data(self):
        binned_data, edges = np.histogramdd([self.energy, self.lon, self.lat], bins=[self.binning.energy_edges, self.binning.lon_edges, self.binning.lat_edges])

        return binned_data, edges

    def get_binned_data(self):
        return self.binned_data
    

    @property
    def binned_energy(self):
        return np.sum(self.binned_data, axis=(1, 2)), self.binning.energy_axis


    @property
    def binned_spatial(self):
        return np.sum(self.binned_data, axis=0), (self.binning.lon_axis, self.binning.lat_axis)

    @property
    def binned_longitude(self):
        return np.sum(self.binned_data, axis=(0, 2)), self.binning.lon_axis


    @property
    def binned_latitude(self):
        return np.sum(self.binned_data, axis=(0, 1)), self.binning.lat_axis
    


    def peek(self, count_scaling='linear', cmap='afmhot',hist1dcolor ='tab:orange', grid=True, grid_kwargs={}, **kwargs):
        
        
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm        
        
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), **kwargs)
        
        axs[0].hist(self.energy_axis, bins=self.binning.energy_edges, weights=self.binned_energy[0], color=hist1dcolor, log=True)
        axs[0].set_xlabel(r'Energy ['+ self.binning.energy_axis.unit.to_string('latex_inline')+']')
        axs[0].set_ylabel('Counts')
        axs[0].set_xlim(np.array([self.binning.energy_edges[0].value, self.binning.energy_edges[-1].value]))
        axs[0].set_yscale(count_scaling)
        axs[0].set_xscale('log')


        if grid:
            full_grid_kwargs = {'which':'major', 'color':'grey', 'ls':'--', 'alpha':0.2, 'zorder':-100}
            full_grid_kwargs.update(grid_kwargs)
            
            axs[0].grid(**full_grid_kwargs)
        
        im = axs[1].imshow(self.binned_spatial[0].T, origin='lower', aspect='auto', extent=[
            self.binning.lon_edges[0].value, 
            self.binning.lon_edges[-1].value,
            self.binning.lat_edges[0].value,
            self.binning.lat_edges[-1].value,
            ], 
            norm=count_scaling,
            cmap=cmap)
        plt.colorbar(im, ax=axs[1], label='Counts')


        axs[1].set_xlabel(r'Longitude ['+self.binning.lon_axis.unit.to_string('latex_inline')+']')
        axs[1].set_ylabel(r'Latitude ['+self.binning.lat_axis.unit.to_string('latex_inline')+']')
        
        fig.tight_layout()

        return fig, axs
    

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
        if self.binning!=other.binning:
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
                        binning=self.binning
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
        return len(self.nonzero_bin_indices[0])
    

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
            current_data = self[self._current_datum_idx]
            self._current_datum_idx += 1
            return current_data
        else:
            raise StopIteration
        
    def to_dict(self):
        return {
            'binning': self.binning.to_dict(),
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
        instance = cls(binning=GammaBinning.from_dict(data['binning']), 
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
        energy_centers = self.binning.energy_axis.value
        lon_centers = self.binning.lon_axis.value
        lat_centers = self.binning.lat_axis.value
        
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
        return np.array(energy_samples) * self.binning.energy_axis.unit, \
            np.array(lon_samples) * self.binning.lon_axis.unit, \
            np.array(lat_samples) * self.binning.lat_axis.unit



class GammaObsCube:
    def __init__(self,
                 binning: GammaBinning, 
                 observations: list[GammaObs],  
                 name:str='NoName', 
                 meta:dict = {}):
        self.name = name
        self.binning = binning
        self.observations = observations
        self.meta = meta

    def add_observation(self, gamma_obs: GammaObs):
        if not np.array_equal(gamma_obs.binning.energy_edges, self.binning.energy_edges) or \
           not np.array_equal(gamma_obs.binning.lon_edges, self.binning.lon_edges) or \
           not np.array_equal(gamma_obs.binning.lat_edges, self.binning.lat_edges):
            raise ValueError("The binning of the observation does not match the binning of the GammaCube.")
        
        self.observations.append(gamma_obs)



    def get_data(self):
        combined_data = sum(obs.get_binned_data() for obs in self.observations)
        return combined_data

    @property
    def energy_axis(self):
        return self.binning.energy_axis
    
    @property
    def lon_axis(self):
        return self.binning.lon_axis
    
    @property
    def lat_axis(self):
        return self.binning.lat_axis

    @property
    def spatial_axes(self):
        return self.lon_axis, self.lat_axis

    @property
    def axes(self):
        return self.energy_axis, self.lon_axis, self.lat_axis

    @property
    def collapsed_energy(self):
        return np.sum(self.get_data(), axis=(1, 2)), self.binning.energy_axis


    @property
    def collapsed_spatial(self):
        return np.sum(self.get_data(), axis=(0)), (self.binning.lon_axis, self.binning.lat_axis)


    @property
    def collapsed_longitude(self):
        return np.sum(self.get_data(), axis=(0, 2)), self.binning.lon_axis



    @property
    def collapsed_latitude(self):
        return np.sum(self.get_data(), axis=(0, 1)), self.binning.lat_axis

    @property
    def collapsed_data(self):
        return self.get_data(), self.binning.axes





    def peek(self, count_scaling='linear', grid=True, grid_kwargs={'which':'major', 'color':'grey', 'ls':'--', 'alpha':0.2, 'zorder':-100}, cmap='afmhot',hist1dcolor ='tab:orange', **kwargs):
        
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm        
        from matplotlib.pyplot import get_cmap

        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), **kwargs)
        
        axs[0].hist(self.energy_axis, bins=self.binning.energy_edges, weights=self.collapsed_energy[0], log=True, color=hist1dcolor)
        axs[0].set_xlabel(r'Energy ['+ self.binning.energy_axis.unit.to_string('latex_inline')+']')
        axs[0].set_ylabel('Counts')
        axs[0].set_xlim(np.array([self.binning.energy_edges[0].value, self.binning.energy_edges[-1].value]))
        axs[0].set_xscale('log')
        axs[0].set_yscale(count_scaling)

        if grid:            
            axs[0].grid(**grid_kwargs)

        im = axs[1].imshow(self.collapsed_spatial[0].T, origin='lower', aspect='auto', extent=[
            self.binning.lon_edges[0].value, 
            self.binning.lon_edges[-1].value,
            self.binning.lat_edges[0].value,
            self.binning.lat_edges[-1].value,
            ], 
            cmap=cmap,
            norm=count_scaling)
        


        plt.colorbar(im, ax=axs[1], label='Counts')
        axs[1].set_xlabel(r'Longitude ['+self.binning.lon_axis.unit.to_string('latex_inline')+']')
        axs[1].set_ylabel(r'Latitude ['+self.binning.lat_axis.unit.to_string('latex_inline')+']')
        axs[1].invert_xaxis()
        
        fig.tight_layout()

        return fig, axs



    def copy(self):
        import copy
        return copy.deepcopy(self)
    

    def __repr__(self):
        return f"GammaCube(name={self.name}, \nbinning={self.binning}, \nobservations={len(self.observations)}, \nmeta={self.meta})"


    
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
                new_obs.append(GammaObs(obs.energy[mask], obs.lon[mask], obs.lat[mask], obs.binning))
        return GammaObsCube(name=self.name, binning=self.binning, observations=new_obs, meta=self.meta)

    @classmethod
    def from_fits(cls, file_name: str):
        # Placeholder for actual implementation
        pass

    @classmethod
    def from_dir(cls, dir_name: str):
        # Placeholder for actual implementation
        pass


    def to_dict(self):
        return {
            'name': self.name,
            'binning': self.binning.to_dict(),
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
        
        binning = GammaBinning.from_dict(data['binning'])
        name = data['name']
        meta = data['meta']
        
        observations = []
        for obs_dict in data['observations']:
            obs = GammaObs(
                binning=GammaBinning.from_dict(obs_dict['binning']),
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



