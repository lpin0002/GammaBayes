import numpy as np
from .binning_geometry import GammaBinning
import pickle
from .exposure import GammaLogExposure
from astropy import units as u
from numpy.typing import ArrayLike
import time
from collections import defaultdict
from icecream import ic

class GammaObs:
    """
    Represents an observation with gamma-ray event data, providing utilities to bin,
    manipulate, and retrieve observation-related data.

    Attributes:
        binning_geometry (GammaBinning): The binning geometry for spatial and energy axes.
        energy (ArrayLike): Energy values of the events.
        lon (ArrayLike): Longitude values of the events.
        lat (ArrayLike): Latitude values of the events.
        pointing_dirs_by_event (np.ndarray): Array of pointing directions for individual events.
        pointing_dirs (np.ndarray): Array of unique pointing directions.
        live_times (u.Quantity): Live times corresponding to pointing directions.
        irf_loglike (callable): Instrument response function log-likelihood.
        log_exposure (GammaLogExposure): Logarithmic exposure map of the observation.
        meta (dict): Metadata for the observation.
    """
    def __init__(self, 
                 binning_geometry: GammaBinning, 
                 name:str = None,
                 energy: ArrayLike =None, lon: ArrayLike=None, lat: ArrayLike=None, 
                 event_weights: ArrayLike = None,
                 pointing_dirs_by_event:np.ndarray[u.Quantity, u.Quantity] = None, 
                 pointing_dirs:np.ndarray[u.Quantity, u.Quantity] = None, 
                 live_times: u.Quantity=None,
                 irf_loglike: callable = None,
                 log_exposure: GammaLogExposure = None,
                 event_ids: np.ndarray = None,
                 meta:dict = None, **kwargs):
        """
        Initializes the GammaObs instance.

        Args:
            binning_geometry (GammaBinning): Binning geometry for the observation.
            name (str, optional): Observation name. Defaults to None.
            energy (ArrayLike, optional): Event energy values. Defaults to None.
            lon (ArrayLike, optional): Event longitude values. Defaults to None.
            lat (ArrayLike, optional): Event latitude values. Defaults to None.
            event_weights (ArrayLike, optional): Weights for events. Defaults to None.
            pointing_dirs_by_event (np.ndarray, optional): Pointing directions per event. Defaults to None.
            pointing_dirs (np.ndarray, optional): Unique pointing directions. Defaults to None.
            live_times (u.Quantity, optional): Live times for pointing directions. Defaults to None.
            irf_loglike (callable, optional): Instrument response function likelihood. Defaults to None.
            log_exposure (GammaLogExposure, optional): Log exposure data. Defaults to None.
            event_ids (np.ndarray, optional): Event IDs. Defaults to None.
            meta (dict, optional): Metadata for the observation. Defaults to None.
            kwargs: Additional metadata key-value pairs.
        """
        
        if meta is None:
            meta = {}

        self.binning_geometry = binning_geometry

        self.energy, self.lon, self.lat = self.__parse_coord_inputs(energy, lon, lat)

        if event_weights is not None:
            energy, lon, lat = self._recreate_samples_with_event_weights(energy, lon, lat, event_weights)



        self.meta = meta

        self.meta.update(kwargs)

        self.__parse_pointing_and_livetime_inputs(pointing_dirs=pointing_dirs,
                                                  pointing_dirs_by_event=pointing_dirs_by_event,
                                                  live_times=live_times)


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


        if 'hemisphere' in self.meta:
            self.hemisphere = self.meta['hemisphere']

        else:
            self.hemisphere = None


        if name is None:
            self.name = time.strftime(f"Observation_pt_%H%M%S")
        else:
            self.name = name


        if len(self.energy)>0 and not(self.irf_loglike=={}):
            self.log_exposure = GammaLogExposure(binning_geometry=self.binning_geometry,
                                            log_exposure_map=log_exposure,
                                            irfs=self.irf_loglike,
                                            pointing_dirs=self.unique_pointing_dir_data,
                                            live_times=self.unique_pointing_live_times)
        elif not(self.irf_loglike=={}):
            self.log_exposure = GammaLogExposure(binning_geometry=self.binning_geometry,
                                            log_exposure_map=log_exposure,
                                            irfs=self.irf_loglike,
                                            pointing_dirs=[],
                                            live_times=[])
        else:
            self.log_exposure = None
            
            
        if len(self.energy):
            self.refresh()

        

    def __parse_coord_inputs(self, energies, longitudes, latitudes):
        """
        Parses and ensures consistency of input coordinate arrays.

        Args:
            energies (ArrayLike): Energy values.
            longitudes (ArrayLike): Longitude values.
            latitudes (ArrayLike): Latitude values.

        Returns:
            tuple: Arrays of parsed energy, longitude, and latitude values.
        """
        if energies is None:
            return [], [], []
        
        try:
            energies = np.array([energ.value for energ in energies])*energies[0].unit
            longitudes = np.array([long.value for long in longitudes])*longitudes[0].unit
            latitudes = np.array([lat.value for lat in latitudes])*latitudes[0].unit

            return energies, longitudes, latitudes
        except:
            return energies, longitudes, latitudes
        
    
    def __parse_pointing_and_livetime_inputs(self, pointing_dirs, live_times, pointing_dirs_by_event):
        """
        Parses and ensures consistency of pointing directions and live times.

        Args:
            pointing_dirs (ArrayLike): Unique pointing directions.
            live_times (u.Quantity): Live times.
            pointing_dirs_by_event (ArrayLike): Pointing directions per event.
        """
        self.live_times = live_times



        if self.live_times is None:
            self.live_times =  self.meta.get('live_times', None)

        if not(hasattr(self.live_times, "unit")) and not(self.live_times is None):
            if hasattr(self.live_times[0], "unit"):
                self.live_times = np.array([live_time.value for live_time in self.live_times])*self.live_times[0].unit
            else:
                self.live_times = np.array(self.live_times)*u.s

        if not(pointing_dirs is None):
            self.pointing_dirs = pointing_dirs

            if not hasattr(self.pointing_dirs, "unit"):
                self.pointing_dirs*=u.deg

        
        elif pointing_dirs is None:
            self.pointing_dirs = self.meta.get('pointing_dirs')

        if self.pointing_dirs is None:
            # If pointing direction not given, the centre of the spatial axes is presumed and
            self.pointing_dirs = np.array([
                np.mean(self.binning_geometry.lon_axis.value), 
                np.mean(self.binning_geometry.lat_axis.to(self.binning_geometry.lon_axis.unit).value)
                ])*self.binning_geometry.lon_axis.unit
        
        if pointing_dirs_by_event is None and self.pointing_dirs.size<3:
            self.pointing_dirs_by_event = np.squeeze(self.pointing_dirs.value)*self.pointing_dirs.unit
        elif pointing_dirs_by_event is None:
            self.pointing_dirs_by_event = self.binning_geometry.spatial_centre
        else:
            self.pointing_dirs_by_event = pointing_dirs_by_event


        if np.array(self.pointing_dirs_by_event).size<3:
            self.pointing_dirs = [self.pointing_dirs_by_event]*len(self.energy)


        if self.live_times is None:
            self.live_times = [1*u.s]*len(self.energy)



    @property
    def energy_axis(self):
        """Returns the energy axis of the binning geometry."""
        return self.binning_geometry.energy_axis
    
    @property
    def lon_axis(self):
        """Returns the longitude axis of the binning geometry."""
        return self.binning_geometry.lon_axis
    
    @property
    def lat_axis(self):
        """Returns the latitude axis of the binning geometry."""
        return self.binning_geometry.lat_axis

    @property
    def spatial_axes(self):
        """Returns the spatial axes (longitude, latitude) of the binning geometry."""
        return self.lon_axis, self.lat_axis

    @property
    def axes(self):
        """Returns all axes (energy, longitude, latitude) of the binning geometry."""
        return self.energy_axis, self.lon_axis, self.lat_axis


    def _bin_data(self):
        """
        Bins the event data into a histogram.

        Returns:
            tuple: Binned data and bin edges.
        """
        binned_data, edges = np.histogramdd([self.energy, self.lon, self.lat], bins=[self.binning_geometry.energy_edges, self.binning_geometry.lon_edges, self.binning_geometry.lat_edges])

        return binned_data, edges    


    def peek(self, axs=None, count_scaling='linear', cmap='afmhot', hist1dcolor='tab:orange', grid=True, grid_kwargs={}, wspace=0.3, **kwargs):
        """
        Visualizes the observation data with multiple plots for energy, spatial, and pointing directions.

        Args:
            axs (matplotlib Axes, optional): Axes to use for plotting. If None, new axes are created. Defaults to None.
            count_scaling (str, optional): Scaling for counts ('linear' or 'log'). Defaults to 'linear'.
            cmap (str, optional): Colormap for 2D histograms. Defaults to 'afmhot'.
            hist1dcolor (str, optional): Color for 1D histograms. Defaults to 'tab:orange'.
            grid (bool, optional): Whether to add a grid to the plots. Defaults to True.
            grid_kwargs (dict, optional): Keyword arguments for grid customization. Defaults to {}.
            wspace (float, optional): Width space between subplots. Defaults to 0.3.
            **kwargs: Additional arguments passed to `plt.subplots`.

        Returns:
            list: List of axes used for plotting.

        Raises:
            ValueError: If `axs` is not None, a single axis, or a list of axes.
        """
        
        
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm        
        import numpy as np


        # Check if axes are provided
        if axs is None:
            # Create a 2x2 subplot grid if no axes are provided
            fig, axs = plt.subplots(2, 2, **kwargs)
        elif isinstance(axs, plt.Axes):
            # If a single axis is passed, replace it with a 2x2 grid of subplots
            fig = axs.figure

            # Get the original grid spec of the axis
            gridspec = axs.get_subplotspec().get_gridspec()
            subplot_spec = axs.get_subplotspec()

            # Remove the original single axis
            fig.delaxes(axs)

            # Create four new subplots in the same grid location with some separation
            gs = gridspec[subplot_spec.rowspan, subplot_spec.colspan].subgridspec(2, 2, wspace=wspace, hspace=0.3)
            axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        elif isinstance(axs, (list, np.ndarray)):
            # Ensure axs are in list/array form
            pass
        else:
            raise ValueError("axs must be either None, a single matplotlib axis, or a list of axes")
        axs = axs.flatten()


        binned_energy, binned_longitudes, binned_latitudes = self.binned_unique_coordinate_data[0]
        event_weights = self.binned_unique_coordinate_data[1]

        if count_scaling == 'linear':
            logq=False
        else:
            logq=True
        

        axs[0].hist(self.energy, bins=self.binning_geometry.energy_edges, color=hist1dcolor, log=logq)
        axs[0].set_xlabel(r'Energy ['+ self.binning_geometry.energy_axis.unit.to_string('latex_inline')+']')
        axs[0].set_ylabel('Counts')
        axs[0].set_xlim(np.array([self.binning_geometry.energy_edges[0].value, self.binning_geometry.energy_edges[-1].value]))
        axs[0].set_yscale(count_scaling)
        axs[0].set_xscale('log')

        


        if grid:
            full_grid_kwargs = {'which':'major', 'color':'grey', 'ls':'--', 'alpha':0.2, 'zorder':-100}
            full_grid_kwargs.update(grid_kwargs)
            
            axs[0].grid(**full_grid_kwargs)
        
        spatial_hist2d_output = axs[1].hist2d(
            self.lon.value, self.lat.value, 
            bins=[self.binning_geometry.lon_axis.value, self.binning_geometry.lat_axis.value],
            norm=count_scaling, cmap=cmap)
        plt.colorbar(spatial_hist2d_output[3], ax=axs[1], label='Counts')


        axs[1].set_xlabel(r'Longitude ['+self.binning_geometry.lon_axis.unit.to_string('latex_inline')+']')
        axs[1].set_ylabel(r'Latitude ['+self.binning_geometry.lat_axis.unit.to_string('latex_inline')+']')
        axs[1].set_aspect('equal')
        axs[1].invert_xaxis()

        # Plot longitude histogram
        axs[2].hist(self.live_times, bins=len(self.binning_geometry.lon_axis), color=hist1dcolor, log=logq)
        axs[2].set_xlabel(r'Live time for pointings [' + self.live_times[0].unit.to_string('latex_inline') + ']')
        axs[2].set_ylabel('Counts')
        axs[2].set_yscale(count_scaling)


        # Plot latitude histogram
        pointing_dirs = np.array(self.pointing_dirs)
        pointing_hist2d_output = axs[3].hist2d(pointing_dirs[:,0], pointing_dirs[:,1], 
                      bins=[len(self.binning_geometry.lon_axis), len(self.binning_geometry.lat_axis)], 
                      norm=count_scaling, cmap=cmap)
        
        axs[3].set_xlabel(r'Pointing Longitude [' + self.binning_geometry.lon_axis.unit.to_string('latex_inline') + ']')
        axs[3].set_ylabel(r'Pointing Latitude [' + self.binning_geometry.lat_axis.unit.to_string('latex_inline') + ']')
        axs[3].set_aspect('equal')
        axs[3].invert_xaxis()
        plt.colorbar(pointing_hist2d_output[3], ax=axs[3], label='Counts')



        if grid:
            full_grid_kwargs = {'which':'major', 'color':'grey', 'ls':'--', 'alpha':0.2, 'zorder':-100}
            full_grid_kwargs.update(grid_kwargs)
            
            axs[0].grid(**full_grid_kwargs)
            axs[2].grid(**full_grid_kwargs)

        # Apply tight_layout to the figure associated with the axs
        axs[0].figure.tight_layout()

        return axs
    


    @property
    def nonzero_bin_data(self):
        """
        Retrieves binned data for non-zero bins, excluding event weights.

        Returns:
            tuple: A tuple containing unique binned coordinate data (energy, longitude, latitude) and event weights.
        """
        
        binned_unique_coordinate_data = self.binned_unique_coordinate_data
        return binned_unique_coordinate_data[:-1], binned_unique_coordinate_data[-1]


    @property
    def _binned_unique_data(self):
        """
        Bins the energy, longitude, and latitude data and computes unique combinations of coordinates.

        Returns:
            tuple: Contains unique energy values, longitude values, latitude values, pointing directions,
                   corresponding live times, and event weights for each unique bin.
        """
        
        # Bin data1, data2, and data3 based on edges
        binned_data1 = np.digitize(self.energy.value, self.binning_geometry.energy_edges.to(self.energy.unit).value) - 1
        binned_data2 = np.digitize(self.lon.value, self.binning_geometry.lon_edges.to(self.lon.unit).value) - 1

        binned_data3 = np.digitize(self.lat.value, self.binning_geometry.lat_edges.to(self.lat.unit).value) - 1

        

        try:
            live_times = np.array(self.live_times.value)
        except AttributeError:
            live_times = np.array([live_time.value for live_time in self.live_times])
        except AttributeError:
            live_times = np.array(self.live_times)


        live_times = live_times*(self.live_times[0].unit)



        # Combine binned data with unbinned data4 to form unique identifiers
        combined_data = list(zip(binned_data1, binned_data2, binned_data3, *np.array(self.pointing_dirs).T))

        # Step 4: Find unique combinations, counts, and indices
        unique_combinations = defaultdict(list)
        for idx, entry in enumerate(combined_data):

            formatted_entry = (self.binning_geometry.energy_axis[entry[0]],self.binning_geometry.lon_axis[entry[1]], self.binning_geometry.lat_axis[entry[2]], entry[-2], entry[-1])
            unique_combinations[formatted_entry].append(idx)


        unique_energy_vals = np.array([datum[0].value for datum in unique_combinations.keys()])*self.energy_axis.unit
        unique_lon_vals = np.array([datum[1].value for datum in unique_combinations.keys()])*self.lon_axis.unit
        unique_lat_vals = np.array([datum[2].value for datum in unique_combinations.keys()])*self.lat_axis.unit

        unique_pointing_dirs = np.array([np.array([datum[3], datum[4]]) for datum in unique_combinations.keys()])*self.pointing_dirs[0][0].unit

        corresponding_live_times = np.array([np.sum(live_times[unique_entry_indices].value) for unique_entry_indices in unique_combinations.values()])*self.live_times[0].unit

        event_weights = [len(unique_entry_indices) for unique_entry_indices in unique_combinations.values()]

        self._unique_energy_data = unique_energy_vals
        self._unique_lon_data = unique_energy_vals
        self._unique_lat_data = unique_energy_vals
        self._unique_pointing_dir_data = unique_pointing_dirs
        self._unique_live_times_for_pt_data = corresponding_live_times
        self._unique_data_event_weights = event_weights

        self._stored_unique_data = unique_energy_vals, unique_lon_vals, unique_lat_vals, unique_pointing_dirs, corresponding_live_times, event_weights

        
        return self._stored_unique_data
    
    def refresh(self):
        """
        Recomputes the unique binned data.

        Returns:
            tuple: Updated binned unique data.
        """
        return self._binned_unique_data
    
    @property
    def binned_unique_coordinate_data(self):
        """
        Retrieves unique binned coordinate data along with event weights.

        Returns:
            tuple: A tuple containing unique binned coordinate data (energy, longitude, latitude) and event weights.
        """
        
        if not hasattr(self, "_stored_unique_data"):
            binned_unique_data = self._binned_unique_data()
            return binned_unique_data[:3], binned_unique_data[-1]

        return self._stored_unique_data[:3], self._stored_unique_data[-1]
    
    @property
    def unique_pointing_dir_data(self):
        """
        Retrieves unique pointing direction data.

        Returns:
            ndarray: Unique pointing directions for the observation.
        """
        if not hasattr(self, "_stored_unique_data"):
            self._binned_unique_data

        return self._unique_pointing_dir_data

    @property
    def unique_pointing_live_times(self):
        """
        Retrieves unique live times for pointing directions.

        Returns:
            ndarray: Unique live times corresponding to pointing directions.
        """
        if not hasattr(self, "_stored_unique_data"):
            self._binned_unique_data

        return self._unique_live_times_for_pt_data
    

    @property
    def unique_energy_data(self):
        """
        Retrieves unique energy data for the observation.

        Returns:
            ndarray: Unique energy values in the observation.
        """
        if not hasattr(self, "_stored_unique_data"):
            self._binned_unique_data

        return self._unique_energy_data


    def __add__(self, other):
        """
        Enables the addition of two GammaObs objects using the '+' operator, concatenating their data arrays.

        Args:
            other (GammaObs): Another GammaObs object to be added to this one.

        Returns:
            GammaObs: A new GammaObs instance containing the combined data of both operands.

        Raises:
            ValueError: If the binning geometries of the two objects do not match.
            NotImplemented: If 'other' is not an instance of GammaObs.
        """
        if not isinstance(other, GammaObs):
            return NotImplemented
        
        if self.binning_geometry!=other.binning_geometry:
            raise ValueError("""When combining multiple sets of data into a single observation, 
it is assumed that the binning geometries are the same.""")


        # Concatenate the attributes of the two instances
        new_energy  = list(self.energy) +list(other.energy)
        new_lon     = list(self.lon) + list(other.lon)
        new_lat     = list(self.lat) + list(other.lat)
        new_pointing_dirs   = list(self.pointing_dirs) + list(other.pointing_dirs)
        new_live_times      = list(self.live_times) + list(other.live_times)

        try:
            if self.log_exposure is None:
                new_log_exposure = other.log_exposure
            elif other.log_exposure is None:
                new_log_exposure = self.log_exposure
            else:
                new_log_exposure = self.log_exposure+other.log_exposure
        except:
            new_log_exposure = np.logaddexp(self.log_exposure.log_exposure_map, 
                                            other.log_exposure.log_exposure_map)

        # You might need to decide how to handle other attributes like obs_id, zenith_angle, etc.
        # For this example, I'm just taking them from the first instance.
        return GammaObs(energy=new_energy, lon=new_lon, lat=new_lat, 
                        pointing_dirs=new_pointing_dirs, 
                        live_times=new_live_times,
                        irf_loglike=self.irf_loglike,
                        binning_geometry=self.binning_geometry,
                        log_exposure=new_log_exposure
)
            
    def __radd__(self, other):
        """
        Supports reverse addition for the '+' operator.

        Args:
            other: If other is 0, it returns self (useful for sum functions).

        Returns:
            GammaObs: Combined GammaObs instance if other is also a GammaObs.
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)
        
    def __len__(self):
        """
        Enables the use of the len() function on a GammaObs instance to obtain the number of events.

        Returns:
            int: The total number of events in the dataset, equivalent to Nevents.
        """
        return len(self.unique_energy_data)
    

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
        Enables indexing into the GammaObs instance like a list or array, allowing direct access to 
        the data array's elements.

        Args:
            key (int, slice): The index or slice of the data array to retrieve.

        Returns:
            list: The data at the specified index or slice for each binned coordinate.
        """
        bin_data, _ = self.binned_unique_coordinate_data
        return [bin_datum[key] for bin_datum in bin_data]
    


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
            list: The next data item in the sequence.
        """
        if self._current_datum_idx < len(self):
            current_data = self[self._current_datum_idx]
            self._current_datum_idx += 1
            return current_data
        else:
            raise StopIteration
        
    def to_dict(self, include_irf=False, include_meta=False):
        """
        Converts the GammaObs instance into a dictionary representation.

        Args:
            include_irf (bool): Whether to include IRF-related information in the dictionary. Defaults to False.
            include_meta (bool): Whether to include metadata in the dictionary. Defaults to False.

        Returns:
            dict: A dictionary representation of the GammaObs instance.
        """

        if hasattr(self.pointing_dirs, "value"):
            pointing_dirs_value = self.pointing_dirs.value
            pointing_dirs_unit = self.pointing_dirs.unit.to_string()
        else:
            pointing_dirs_value = self.pointing_dirs
            pointing_dirs_unit = None

        if hasattr(self.live_times, "value"):
            live_times_value = self.live_times.value
            live_times_unit = self.live_times.unit.to_string()
        else:
            live_times_value = self.live_times
            live_times_unit = None


        output_dict = {
            'name': self.name,
            'binned_data': self.binned_data,
            'pointing_dirs': pointing_dirs_value,
            'pointing_dirs_unit': pointing_dirs_unit,
            'live_times': live_times_value,
            'live_times_unit': live_times_unit,
        }
        if include_irf:
            output_dict['irf_loglike'] = self.irf_loglike

        if include_meta:
            output_dict['meta'] = self.meta
            output_dict['meta']['binning_geometry'] = self.binning_geometry.to_dict()


        return output_dict


    def save(self, filename, save_meta=False):
        """
        Saves the GammaObs instance to a file.

        Args:
            filename (str): The name of the file to save the data to.
            save_meta (bool): Whether to include metadata in the saved file. Defaults to False.
        """
        data_to_save = self.to_dict(include_meta=save_meta)
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)


    @classmethod
    def load_from_dict(cls, info_dict, aux_info={}):
        """
        Creates a GammaObs instance from a dictionary representation.

        Args:
            info_dict (dict): The dictionary containing the GammaObs data.
            aux_info (dict): Additional auxiliary information.

        Returns:
            GammaObs: A new GammaObs instance created from the dictionary.
        """
        binning_geometry_data = info_dict.get('binning_geometry')

        if binning_geometry_data is None:
            binning_geometry = aux_info.get("binning_geometry")
        elif isinstance(binning_geometry_data, dict):
            binning_geometry = GammaBinning.from_dict(binning_geometry_data)
        else:
            binning_geometry = binning_geometry_data
        name = info_dict.get("name")
        if name is None:
            name = aux_info.get("name")



        pointing_dirs = info_dict.get('pointing_dirs')
        if (info_dict.get('pointing_dirs_unit') is not None):
            pointing_dirs = pointing_dirs*u.Unit(info_dict.get('pointing_dirs_unit'))
        elif isinstance(pointing_dirs, ArrayLike): 
            if hasattr(pointing_dirs, "unit"):
                pointing_dirs_value = pointing_dirs.value
            else:
                pointing_dirs_value = pointing_dirs
            if np.isnan(np.sum(pointing_dirs_value)): # To account for saving variables as nan for h5 files
                pointing_dirs = None


        live_times = info_dict.get('live_times')
        if not hasattr(live_times, "unit"):
            if not(isinstance(live_times, float)):
                live_times = None
            
        if (info_dict.get('live_times_unit') is not None) and not(np.isnan(info_dict.get('live_times_unit'))):
            live_times = live_times*u.Unit(info_dict.get('live_times_unit'))

        try:
            live_times = u.Quantity(live_times)
        except:
            live_times = None


        # Create an empty instance
        instance = cls(binning_geometry=binning_geometry, 
                    name = name,
                    energy=[], lon=[], lat=[],  # Empty raw samples
                    meta=info_dict.get('meta', {}),
                    pointing_dirs=pointing_dirs,
                    live_times=live_times,
                    irf_loglike=info_dict.get('irf_loglike'))
        
        # Load the binned data
        instance.binned_data = info_dict['binned_data']
        
        # Recreate the samples from the binned data
        energy_samples, lon_samples, lat_samples = instance._recreate_samples_from_binned_data()
        instance.energy = energy_samples
        instance.lon = lon_samples
        instance.lat = lat_samples
        
        return instance
    

    @classmethod
    def load(cls, filename, aux_info: dict):
        """
        Loads a GammaObs instance from a saved file.

        Args:
            filename (str): The name of the file to load data from.
            aux_info (dict): Additional auxiliary information.

        Returns:
            GammaObs: A new GammaObs instance created from the file.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        return cls.load_from_dict(data)
            




    def _recreate_samples_from_binned_data(self):
        """
        Recreates raw samples of energy, longitude, and latitude from binned data.

        Returns:
            tuple: Arrays of energy, longitude, and latitude samples.
        """
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
    

    def _recreate_samples_with_event_weights(self, energy_vals, lon_vals, lat_vals, event_weights):
        """
        Recreates raw samples of energy, longitude, and latitude by applying event weights.

        Args:
            energy_vals (Quantity): Energy values of the events.
            lon_vals (Quantity): Longitude values of the events.
            lat_vals (Quantity): Latitude values of the events.
            event_weights (ndarray): Weights for each event.

        Returns:
            tuple: Weighted arrays of energy, longitude, and latitude samples.
        """
        original_energy_values = energy_vals.value
        original_energy_unit = energy_vals.unit

        original_lon_values = lon_vals.value
        original_lon_unit = lon_vals.unit

        original_lat_values = lat_vals.value
        original_lat_unit = lat_vals.unit

        # Prepare lists to hold the recreated samples
        energy_samples = []
        lon_samples = []
        lat_samples = []
        
        # Loop over the binned data to recreate the samples
        for event_weight, energy, longitude, latitude  in zip(event_weights, original_energy_values, original_lon_values, original_lat_values):
            # Repeat the bin centers according to the bin count
            energy_samples.extend([energy] * int(event_weight))
            lon_samples.extend([longitude] * int(event_weight))
            lat_samples.extend([latitude] * int(event_weight))
        
        # Convert to numpy arrays and return
        return np.array(energy_samples) * original_energy_unit, \
            np.array(lon_samples) * original_lon_unit, \
            np.array(lat_samples) * original_lat_unit



    @property
    def obs_meta(self):
        """
        Generates metadata for each observation, including pointing directions and live times.

        Returns:
            list: A list of dictionaries containing metadata for each observation.
        """
        __meta_list = []

        for pointing_dir, live_time in zip(self.pointing_dirs, self.live_times):
            __meta_list.append({'pointing_dir':pointing_dir, 'live_time':live_time})

        return __meta_list

