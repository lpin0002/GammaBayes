from matplotlib import pyplot as plt, colors
import h5py, time, numpy as np, warnings
from gammabayes.core.core_utils import bin_centres_to_edges

class EventData(object):

    def __init__(self, 
                 energy: np.ndarray = None, 
                 glon: np.ndarray = None, 
                 glat: np.ndarray = None, 
                 data: np.ndarray = None, 
                 event_times: np.ndarray = None, 
                 pointing_dirs: np.ndarray = None, 
                 obs_start_time: float = None, 
                 obs_end_time: float = None,
                 hemisphere: str = None, 
                 zenith_angle: int = None, 
                 obs_id: any ='NoID',
                 energy_axis: np.ndarray = np.nan, 
                 glongitude_axis: np.ndarray = np.nan, 
                 glatitude_axis: np.ndarray = np.nan,
                 _source_ids: np.ndarray[str] = None, 
                 _likelihood_id: np.ndarray[str] = np.nan, 
                 _true_vals: bool = False):
        """
        Initializes an EventData object to manage and preprocess astronomical event data.
        
        This class is designed to handle event data for astronomical observations, 
        including energy measurements, galactic longitude (glon), and galactic latitude (glat). 
        It supports initializing data either from separate arrays of energy, glon, and glat, 
        or a combined data array. Additional data regarding the observation can also be provided.
        
        Parameters:
        - energy (array-like, optional): Array of energy measurements for the events.
        
        - glon (array-like, optional): Array of galactic longitude coordinates for the events.
        
        - glat (array-like, optional): Array of galactic latitude coordinates for the events.
        
        - data (array-like, optional): Combined array of the format [energy, glon, glat, ...] for the events.
        
        - event_times (array-like, optional): Array of observation times for the events.
        
        - pointing_dirs (array-like, optional): Array of pointing directions for the telescope during observations.
        
        - obs_start_time, obs_end_time (scalar, optional): Start and end times of the observation period.
        
        - hemisphere (str, optional): Hemisphere in which the observation was made.
        
        - zenith_angle (int, optional): Zenith angle of the observation.
        
        - obs_id (str, optional): Identifier for the observation, defaults to 'NoID'.
        
        - energy_axis, glongitude_axis, glatitude_axis (array-like, optional): Axes definitions for binning the data. Defaults to np.nan, indicating automatic binning.
        
        - _source_ids (array-like, optional): Private attribute for source IDs.
        
        - _likelihood_id (scalar, optional): Private attribute for likelihood ID, defaults to np.nan.
        
        - _true_vals (bool, optional): Flag to indicate if true values are being used, defaults to False.
        
        Attributes:
        - energy, glon, glat: Arrays containing the energy, galactic longitude, and latitude of events.
        
        - data: Structured array containing the event data.
        
        - event_times: Array of event times.
        
        - pointing_dirs: Array of pointing directions.
        
        - obs_id: Observation ID.
        
        - hemisphere: Hemisphere of observation.
        
        - zenith_angle: Zenith angle of observation.
        
        - obs_start_time, obs_end_time: Observation start and end times.
        
        - energy_axis, glongitude_axis, glatitude_axis: Axes for data binning.
        
        - _source_ids: Source IDs (private).
        
        - _likelihood_id: Likelihood ID (private).
        
        - _true_vals: Indicates if true values are used (private).
        
        The constructor handles missing data by assigning NaNs. 
        It defines axes and bins for energy and angular coordinates, falling back to default bins if not provided. 
        Observation metadata such as ID, hemisphere, zenith angle, and start/end times are stored, along with 
        private attributes for source and likelihood ID.
        """

        

        # Initialize data arrays based on provided energy, glon, and glat, or from 'data' array
        # Handle missing data with NaNs and setup array shapes appropriately

        # Define axes and bins for energy and angular coordinates
        # Fallback to default bins if not provided

        # Store observation metadata like ID, hemisphere, zenith angle, and start/end times
        # Private attributes for source and likelihood ID

        
        if not(energy is None):

            self.energy             = np.asarray(energy)
            self.glon               = np.asarray(glon)
            self.glat               = np.asarray(glat)

            self.data = np.asarray([*zip(self.energy, self.glon, self.glat)])

        
        elif not(data is None):
            self.data = np.asarray(data)

            self.energy             = np.asarray(self.data[:,0])
            self.glon               = np.asarray(self.data[:,1])
            self.glat               = np.asarray(self.data[:,2])


        else:
            warnings.warn("No energy, longitude or latitude values given. Assigning empty lists.")
            self.energy             = np.asarray([])
            self.glon               = np.asarray([])
            self.glat               = np.asarray([])
            self.data = np.asarray([*zip(self.energy, self.glon, self.glat)])



        try:
            self.data.shape[1]
            data_shape = self.data.shape
        except IndexError:
            data_shape = (len(self.energy),3)

        if not(pointing_dirs is None):
            self.pointing_dirs  = pointing_dirs
        elif data_shape[1]>3:
            self.pointing_dirs  = self.data[:,3]
        else:
            self.pointing_dirs  = np.full(shape=(len(self.energy),2), fill_value=[np.nan, np.nan])



        if not(event_times is None):
            self.event_times  = event_times
        elif data_shape[1]>4:
            self.event_times  = self.data[:,4]
        else:
            self.event_times  = np.full(shape=(len(self.energy),), fill_value=np.nan)



        if not(_source_ids is None):
            self._source_ids  =  _source_ids
        elif data_shape[1]>5:
            self._source_ids  = self.data[:,5]
        else:
            self._source_ids  = np.full(shape=(len(self.energy),), fill_value=np.nan)


        if self.data.ndim>1:
            self.data = self.data[:, :3]
        else:
            if len(self.data)>0:
                self.data = np.asarray([self.data[:3]])
            else:
                self.data = np.array([[], [], []]).T


        self.energy_axis      = energy_axis
        
        try:
            len(self.energy_axis)
            self.energy_bins        = bin_centres_to_edges(energy_axis)
        except:
            self.energy_bins        = np.logspace(-1.5,2.5,81)

        self.glongitude_axis  = glongitude_axis
        self.glatitude_axis   = glatitude_axis


        try:
            len(self.glongitude_axis)
            self.angular_bins        = bin_centres_to_edges(glongitude_axis)
        except:
            self.angular_bins        = 101


        self.obs_id             = obs_id
        self.hemisphere         = hemisphere
        self.zenith_angle       = zenith_angle
        self.obs_start_time     = obs_start_time
        self.obs_end_time       = obs_end_time

        self._likelihood_id     = _likelihood_id
        self._true_vals         = _true_vals


    

    def __len__(self):
        """
        Enables the use of the len() function on an EventData instance to obtain the number of events.

        Returns:
            int: The total number of events in the dataset, equivalent to Nevents.
        """
        return self.Nevents
    
    # Property to return the number of events
    @property
    def Nevents(self):
        """
        Calculates the number of events by counting the elements in the energy array.

        Returns:
            int: The total number of events in the dataset.
        """
        return len(self.energy)
    

    # Property to return the 'full' data set as a list of tuples
    @property
    def full_data(self):
        """
        Constructs the 'full' dataset as a list of tuples, each containing data for a single event,
        including energy, galactic longitude, galactic latitude, pointing directions, and event times.

        Returns:
            list of tuples: The complete dataset with each tuple representing an event's data.
        """
        return [*zip(self.energy, self.glon, self.glat, self.pointing_dirs, self.event_times)]
    
    # Support for indexing like a list or array
    def __getitem__(self, key: int | slice):
        """
        Enables indexing into the EventData instance like a list or array, allowing direct access to 
        the data array's elements.

        Args:
            key (int, slice): The index or slice of the data array to retrieve.

        Returns:
            ndarray: The data at the specified index or slice.
        """
        return self.data[key]
    


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
        if self._current_datum_idx < len(self.data):
            current_data = self.data[self._current_datum_idx]
            self._current_datum_idx += 1
            return current_data
        else:
            raise StopIteration
        
    # String representation for the EventData object
    def __str__(self):
        """
        Provides a string representation of the EventData object, summarizing its key attributes and statistics.

        This method formats the observation data and statistics into a human-readable summary, 
        including the number of events, observation ID, hemisphere, zenith angle, observation start and end times, 
        total observation time, minimum and maximum event times, number of angular and energy bins, 
        minimum and maximum energy, and information on source and likelihood IDs.

        Returns:
            str: A formatted string summarizing the EventData object's key attributes and statistics.
        """
        num_spaces = 31
        strinfo = "\n"
        strinfo += "="*(num_spaces+20)+"\n"
        strinfo += f"Event Data ID: {self.obs_id}\n"
        strinfo += ":"*num_spaces+"\n"
        strinfo += f"Nevents                |   {self.Nevents}\n"
        strinfo += f"Hemisphere             |   {self.hemisphere}\n"
        strinfo += f"Zenith Angle           |   {self.zenith_angle}\n"
        strinfo += "-"*num_spaces+"\n"
        strinfo += f"Obs Start Time         |   {self.obs_start_time}\n"
        strinfo += f"Obs End Time           |   {self.obs_end_time}\n"

        try:
            strinfo += f"Total Obs Time         |   {self.obs_end_time-self.obs_start_time} s\n"
        except:
            strinfo += f"Total Obs Time         |   NA\n"

        strinfo += "- "*int(round(num_spaces/2))+"\n"

        try:
            strinfo += f"Min Event Time         |   {min(self.event_times)}\n"
        except TypeError:
            strinfo += f"Min Event Time         |   NA\n"

        try:
            strinfo += f"Max Event Time         |   {max(self.event_times)}\n"
        except TypeError:
            strinfo += f"Max Event Time         |   NA\n"


        strinfo += "-"*num_spaces+"\n"

        strinfo += f"Num Angular Bins       |   {len(self.angular_bins)}\n"

        try:
            strinfo += f"Num Energy Bins        |   {len(self.energy_bins)}\n"
        except TypeError:
            strinfo += f"Num Energy Bins        |   {self.energy_bins}\n"

        strinfo += "- "*int(round(num_spaces/2))+"\n"

        strinfo += f"Min Energy             |   {min(self.energy):.3f} TeV\n"
        strinfo += f"Max Energy             |   {max(self.energy):.3f} TeV\n"

        strinfo += "-"*num_spaces+"\n"

        strinfo += f"Source IDs             |   {np.unique(self._source_ids)}\n"
        strinfo += f"Likelihood ID          |   {self._likelihood_id}\n"
        strinfo += f"True Values?           |   {self._true_vals}\n"

        strinfo += "="*(num_spaces+20)+"\n"

        return strinfo
    



    # Method to update raw data arrays with new data
    def update_raw_data(self, 
                        energy: np.ndarray = None, 
                        glon: np.ndarray = None, 
                        glat: np.ndarray = None, 
                        pointing_dirs=None, 
                        event_times: np.ndarray = None, 
                        _source_ids: np.ndarray = None, 
                        data: np.ndarray = None):
        """
        Updates the raw data arrays of the EventData object with new data. 

        This method allows for the addition of new event data either through individual parameter arrays 
        or a unified data array. It appends new data to existing arrays, updating the EventData object.
        Other parameters besides the energy, glon, glat, pointing_dirs, event_times, _source_ids and data,
        are not expected to be updated as the class is typically for a single observation, and the other
        parameters describe this observation thus should not need be updated.

        Args:
            energy (array-like, optional): New energy measurements to be added.
            
            glon (array-like, optional): New galactic longitude measurements to be added.
            
            glat (array-like, optional): New galactic latitude measurements to be added.
            
            pointing_dirs (array-like, optional): New pointing directions to be added.
            
            event_times (array-like, optional): New event times to be added.
            
            _source_ids (array-like, optional): New source IDs to be added.
            
            data (array-like, optional): Unified array containing new data in the format [energy, glon, glat, ...].
        """

        # Checks to see if energy is given and if the first argument is actually data
        if not(energy is None) and (np.asarray(energy).ndim==1):

            data = np.asarray([*zip(energy, glon, glat)])

            self.energy = np.append(self.energy, np.asarray(energy))
            self.glon   = np.append(self.glon, np.asarray(glon))
            self.glat   = np.append(self.glat, np.asarray(glat))

        # If the first argument is actually data then...
        elif (np.asarray(energy).ndim==2):
            data=np.asarray(energy)

        try:
            appending_data = data[:,:3]
            
        except IndexError:
            if data.ndim<2:
                appending_data = np.empty(shape=(1,len(data)))
                appending_data[0,:] = appending_data


        self.data = np.append(self.data, appending_data[:,:3], axis=0)

        self.energy = self.data[:,0]
        self.glon   = self.data[:,1]
        self.glat   = self.data[:,2]



        if not(pointing_dirs is None):
            self.pointing_dirs  = np.append(self.pointing_dirs, pointing_dirs)
        elif data.shape[1]>3:
            self.pointing_dirs  = np.append(self.pointing_dirs, data[:,3])
        else:
            self.pointing_dirs  = np.append(self.pointing_dirs, np.full(shape=(len(data[:,0]),2), fill_value=[np.nan, np.nan]))



        if not(event_times is None):
            self.event_times  = np.append(self.event_times, event_times)
        elif data.shape[1]>4:
            self.event_times  = np.append(self.event_times, data[:,4])
        else:
            self.event_times  = np.append(self.event_times, np.full(shape=(len(data[:,0]),), fill_value=np.nan))



        if not(_source_ids is None):
            self._source_ids  = np.append(self._source_ids, _source_ids)
        elif data.shape[1]>5:
            self._source_ids  = np.append(self._source_ids, data[:,5])
        else:
            self._source_ids  = np.append(self._source_ids, np.full(shape=(len(data[:,0]),), fill_value=np.nan))


    # Method to append new data to the existing dataset
    def append(self, new_data: np.ndarray):
        """
        Appends new data to the existing dataset.

        This method allows for the addition of new data from another EventData instance or a structured data array.

        Args:
            new_data (EventData or array-like): The new data to be appended. If new_data is an EventData instance, its data is extracted and appended; otherwise, the method assumes new_data is a structured array compatible with the EventData format.
        """
        if type(new_data) is EventData:
            if new_data._true_vals == self._true_vals:
                self.update_raw_data(pointing_dirs=new_data.pointing_dirs, event_times=new_data.event_times, data=new_data.data)
            else:
                raise ValueError("You are trying to append true values to reconstructed values or vice versa.")

        else:
            self.update_raw_data(new_data)


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
        if not isinstance(other, EventData):
            return NotImplemented
        if other._true_vals == self._true_vals:
            # Concatenate the attributes of the two instances
            new_energy = np.concatenate([self.energy, other.energy])
            new_glon = np.concatenate([self.glon, other.glon])
            new_glat = np.concatenate([self.glat, other.glat])
            new_pointing_dirs = np.concatenate([self.pointing_dirs, other.pointing_dirs])
            new_event_times = np.concatenate([self.event_times, other.event_times])
            new_source_ids = np.concatenate([self._source_ids, other._source_ids])


            # You might need to decide how to handle other attributes like obs_id, zenith_angle, etc.
            # For this example, I'm just taking them from the first instance.
            return EventData(energy=new_energy, glon=new_glon, glat=new_glat, 
                            pointing_dirs=new_pointing_dirs, event_times=new_event_times, 
                            _source_ids=new_source_ids,
                            
                            obs_id=self.obs_id, 
                            hemisphere=self.hemisphere, zenith_angle=self.zenith_angle, 

                            obs_start_time=self.obs_start_time, obs_end_time=self.obs_end_time,

                            energy_axis=self.energy_axis, glongitude_axis=self.glongitude_axis, 
                            glatitude_axis=self.glatitude_axis, 

                            _likelihood_id=self._likelihood_id, _true_vals=self._true_vals)
        else:
            raise ValueError("You are trying to add true values to reconstructed values or vice versa.")
        
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    # Extract raw data lists for external use
    def extract_lists(self):
        """
        Extracts raw data arrays as lists for external use.

        Returns:
            tuple: Contains arrays of energy, galactic longitude, galactic latitude, pointing directions, and event times.
        """
        return self.energy, self.glon, self.glat, self.pointing_dirs, self.event_times
    

    # Extract axes information for energy and angular coordinates
    def extract_event_axes(self):
        """
        Extracts the axes information for energy and angular coordinates.

        Returns:
            tuple: Contains arrays for energy axis, galactic longitude axis, and galactic latitude axis.
        """
        return self.energy_axis, self.glongitude_axis, self.glatitude_axis


    # Visualization method to peek at the data distributions
    def peek(self, 
             figsize: tuple[int] = (10,4), 
             scale:str = 'log', 
             *args, 
             **kwargs):
        """
        Generates a visualization of the data distributions for energy and angular coordinates.
        It expects a non-zero number of events. Any arguments or keywords outside figsize and 
        colorscale are passed into 'plt.subplots'.

        Args:
            figsize (tuple, optional): Size of the figure.
            
            colorscale (str, optional): Color scale to use for the histogram. Defaults to 'log' for logarithmic color scaling.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axes objects of the plot.
        """
        if scale == 'log':
            colorscale = colors.LogNorm()
        else:
            colorscale=scale


        fig, ax = plt.subplots(1,2, figsize=figsize, *args, **kwargs)
        ax[0].hist(self.energy, bins=self.energy_bins)
        ax[0].set_xlabel('Energy [TeV]')
        ax[0].set_ylabel('Events')
        ax[0].set_xscale('log')
        ax[0].set_yscale(scale)

        hist2d = ax[1].hist2d(self.glon, self.glat, bins=self.angular_bins, norm=colorscale)
        ax[1].set_xlabel('Galactic Longitude [deg]')
        ax[1].set_ylabel('Galactic Latitude [deg]')
        plt.colorbar(hist2d[3], label='Events')
        
        fig.tight_layout()

        return fig, ax
    
    def hist_sources(self, 
                     figsize: tuple[int] = (6,6), 
                     *args, 
                     **kwargs):
        """
        Creates a histogram of the source IDs to visualize the distribution of events across sources.
        Everything else besides figsize is passed into 'plt.figure'.

        Args:
            figsize (tuple, optional): Size of the figure.
        """
        fig = plt.figure(figsize=figsize, *args, **kwargs)
        _, bins, _ = plt.hist(self._source_ids, bins=len(np.unique(self._source_ids)))
        ax = plt.gca()
        labels = [item.get_text() for item in ax.get_xticklabels()]
        # Ensure the number of labels matches the number of bins

        # Modify labels to wrap text
        wrapped_labels = [label.replace(' ', '\n') for label in labels]

        # Set new labels and rotate
        plt.xticks(bins[:-1]+0.5*np.diff(bins)[0], wrapped_labels, rotation=25)
        plt.ylabel("Events")
        plt.show()


    def create_batches(self, num_batches: int = 10):
        """
        Splits the data into a specified number of batches, each as a separate EventData object.

        Args:
            num_batches (int, optional): The number of batches to create.

        Raises:
            ValueError: If num_batches is not a positive integer or is too large for the dataset size.

        Returns:
            list of EventData: A list containing the split EventData objects.
        """
        if num_batches <= 0 or not isinstance(num_batches, int):
            raise ValueError("Number of batches must be a positive integer.")

        # Calculate the size of each batch
        batch_size = len(self.energy) // num_batches
        if batch_size == 0:
            raise ValueError("Number of batches is too large for the dataset size.")

        batches = []

        for i in range(num_batches):
            # Determine the start and end indices for each batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            # Adjust the last batch to include any remaining data
            if i == num_batches - 1:
                end_idx = len(self.energy)

            # Create a new EventData object for each batch
            batch = EventData(
                energy=self.energy[start_idx:end_idx],
                glon=self.glon[start_idx:end_idx],
                glat=self.glat[start_idx:end_idx],
                event_times=self.event_times[start_idx:end_idx],
                pointing_dirs=self.pointing_dirs[start_idx:end_idx],
                _source_ids=self._source_ids[start_idx:end_idx],
                obs_id=self.obs_id,
                hemisphere=self.hemisphere,
                zenith_angle=self.zenith_angle,
                obs_start_time=self.obs_start_time,
                obs_end_time=self.obs_end_time,
                energy_axis=self.energy_axis,
                glongitude_axis=self.glongitude_axis,
                glatitude_axis=self.glatitude_axis,
                _likelihood_id=self._likelihood_id,
                _true_vals=self._true_vals
            )

            batches.append(batch)

        return batches

    

    # Method to save the current state of the object to an HDF5 file
    def save(self, filename: str = None):
        """
        Saves the current state of the EventData object to an HDF5 file.

        Args:
            filename (str, optional but advised): Name of the file to save the data to. If not provided, a filename is generated based on the observation ID, hemisphere, zenith angle, and current time.
        """
        if filename is None:
            filename = time.strftime(
                f"EventData_ID{self.obs_id}_Hem{self.hemisphere}_Zen{self.zenith_angle}_Events{self.Nevents}_%y_%m_%d_%H_%M_%S.h5")

        if not(filename.endswith(".h5")):
            filename = filename + ".h5"

        # TODO: Check if file exists beforehand and then tack on date stamp to the end if it does
        with h5py.File(filename, 'w-') as f:
            f.create_dataset('energy',          data=self.energy)
            f.create_dataset('glon',            data=self.glon)
            f.create_dataset('glat',            data=self.glat)
            f.create_dataset('event_times',     data=self.event_times)
            f.create_dataset('pointing_dirs',   data=self.pointing_dirs)

            str_dtype = h5py.special_dtype(vlen=str)  # Define a variable-length string data type
            source_ids_dataset = f.create_dataset('_source_ids', (len(self._source_ids),), dtype=str_dtype)
            source_ids_dataset[:] = self._source_ids  # Assign the string data
            f.create_dataset('energy_axis',     data=self.energy_axis)
            f.create_dataset('glongitude_axis', data=self.glongitude_axis)
            f.create_dataset('glatitude_axis',  data=self.glatitude_axis)


            f.attrs['obs_id']                       = self.obs_id

            if self.obs_start_time is not None:
                f.attrs['obs_start_time']               = self.obs_start_time
                f.attrs['obs_end_time']                 = self.obs_end_time
            
            if self.hemisphere is not None:
                f.attrs['hemisphere']                   = self.hemisphere

            if self.zenith_angle is not None:
                f.attrs['zenith_angle']                 = self.zenith_angle

            f.attrs['_likelihood_id']               = self._likelihood_id
            f.attrs['_true_vals']                   = self._true_vals

    # Class method to load an EventData object from an HDF5 file
    @classmethod
    def load(cls, filename: str):
        """
        Loads an EventData object from an HDF5 file.

        Args:
            filename (str): The path to the HDF5 file from which to load the data.

        Returns:
            EventData: An instance of EventData initialized with the data loaded from the specified file.
        """
        if not(filename.endswith(".h5")):
            filename = filename + ".h5"
        
        with h5py.File(filename, 'r') as f:
            # Access the datasets
            energy          = np.array(f['energy'])
            glon            = np.array(f['glon'])
            glat            = np.array(f['glat'])
            event_times     = np.array(f['event_times'])
            pointing_dirs   = np.array(f['pointing_dirs'])
            _source_ids     = np.array(f['_source_ids'])
            # Decode each string from bytes to str if they are bytes
            if _source_ids.dtype == object:  # Checks if the dtype is object, indicating bytes
                _source_ids_decoded = np.array([s.decode('utf-8') if isinstance(s, bytes) else s for s in _source_ids])
            else:
                _source_ids_decoded = _source_ids  # If already strings, no need to decode
            energy_axis     = np.array(f['energy_axis'])
            glongitude_axis = np.array(f['glongitude_axis'])
            glatitude_axis  = np.array(f['glatitude_axis'])

            # Access the attributes (metadata)
            obs_id          = f.attrs['obs_id']
            if 'obs_start_time' in f.attrs:
                obs_start_time  = f.attrs['obs_start_time']
                obs_end_time    = f.attrs['obs_end_time']
            else:
                obs_start_time = None
                obs_end_time = None

            if 'hemisphere' in f.attrs:
                hemisphere      = f.attrs['hemisphere']
            else:
                hemisphere = None
            if 'zenith_angle' in f.attrs:
                zenith_angle    = f.attrs['zenith_angle']
            else:
                zenith_angle = None

            _likelihood_id  = f.attrs['_likelihood_id']
            _true_vals      = f.attrs['_true_vals']


        return EventData(energy=energy, glon=glon, glat=glat, pointing_dirs=pointing_dirs,
                         energy_axis=energy_axis, glongitude_axis=glongitude_axis, glatitude_axis=glatitude_axis,
                         obs_id=obs_id, hemisphere=hemisphere, zenith_angle=zenith_angle,
                         event_times=event_times,
                         obs_start_time=obs_start_time, obs_end_time=obs_end_time,
                         _source_ids=_source_ids_decoded, _likelihood_id=_likelihood_id, _true_vals=_true_vals)



