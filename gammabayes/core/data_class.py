from matplotlib import pyplot as plt, colors
import h5py, time, numpy as np, warnings
from gammabayes.core.core_utils import bin_centres_to_edges

class EventData(object):

    def __init__(self, energy=None, glon=None, glat=None, data=None, 
                 event_times=None, pointing_dirs=None, 
                 obs_start_time=None, obs_end_time=None,
                 hemisphere=None, zenith_angle=None, obs_id='NoID',
                 energy_axis=np.nan, glongitude_axis=np.nan, glatitude_axis=np.nan,
                 _source_ids=None, _likelihood_id=np.nan, _true_vals=False):
        

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


    

    # Property to return the number of events
    @property
    def Nevents(self):
        return len(self.energy)
    

    # Property to return the 'full' data set as a list of tuples
    @property
    def full_data(self):
        return [*zip(self.energy, self.glon, self.glat, self.pointing_dirs, self.event_times)]
    
    # Support for indexing like a list or array
    def __getitem__(self, key):
        return self.data[key]
    

    # Iterator support to enable looping over the dataset
    def __iter__(self):
        self._current_datum_idx = 0  # Reset the index each time iter is called
        return self

    # Next method for iterator protocol
    def __next__(self):
        if self._current_datum_idx < len(self.data):
            current_data = self.data[self._current_datum_idx]
            self._current_datum_idx += 1
            return current_data
        else:
            raise StopIteration
        
    # String representation for the EventData object
    def __str__(self):
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

        strinfo += f"Num Angular Bins       |   {self.angular_bins}\n"

        try:
            strinfo += f"Num Energy Bins        |   {len(self.energy_bins)}\n"
        except TypeError:
            strinfo += f"Num Energy Bins        |   {self.energy_bins}\n"

        strinfo += "- "*int(round(num_spaces/2))+"\n"

        strinfo += f"Min Energy             |   {min(self.energy)} TeV\n"
        strinfo += f"Max Energy             |   {max(self.energy)} TeV\n"

        strinfo += "-"*num_spaces+"\n"

        strinfo += f"Source IDs             |   {np.unique(self._source_ids)}\n"
        strinfo += f"Likelihood ID          |   {self._likelihood_id}\n"
        strinfo += f"True Values?           |   {self._true_vals}\n"

        strinfo += "="*(num_spaces+20)+"\n"

        return strinfo
    



    # Method to update raw data arrays with new data
    def update_raw_data(self, energy=None, glon=None, glat=None, 
                        pointing_dirs=None, event_times=None, _source_ids=None, 
                        data=None):

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
    def append(self, new_data):
        if type(new_data) is EventData:
            self.update_raw_data(pointing_dirs=new_data.pointing_dirs, event_times=new_data.event_times, data=new_data.data)

        else:
            self.update_raw_data(new_data)


    def __add__(self, other):
        if not isinstance(other, EventData):
            return NotImplemented

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

    # Extract raw data lists for external use
    def extract_lists(self):
        return self.energy, self.glon, self.glat, self.pointing_dirs, self.event_times
    

    # Extract axes information for energy and angular coordinates
    def extract_event_axes(self):
        return self.energy_axis, self.glongitude_axis, self.glatitude_axis


    # Visualization method to peek at the data distributions
    def peek(self, figsize=(10,4), colorscale='log', *args, **kwargs):
        if colorscale == 'log':
            colorscale = colors.LogNorm()
        else:
            colorscale = None

        fig, ax = plt.subplots(1,2, figsize=figsize, *args, **kwargs)
        ax[0].hist(self.energy, bins=self.energy_bins)
        ax[0].set_xlabel('Energy [TeV]')
        ax[0].set_ylabel('Events')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')

        hist2d = ax[1].hist2d(self.glon, self.glat, bins=self.angular_bins, norm=colorscale)
        ax[1].set_xlabel('Galactic Longitude [deg]')
        ax[1].set_ylabel('Galactic Latitude [deg]')
        plt.colorbar(hist2d[3], label='Events')
        
        fig.tight_layout()

        return fig, ax
    
    def hist_sources(self, figsize=(6,6), *args, **kwargs):
        fig = plt.figure(figsize=(6,6), *args, **kwargs)
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

    

    # Method to save the current state of the object to an HDF5 file
    def save(self, filename=None):
        if filename is None:
            filename = time.strftime(
                f"EventData_ID{self.obs_id}_Hem{self.hemisphere}_Zen{self.zenith_angle}_Events{self.Nevents}_%y_%m_%d_%H_%M_%S.h5")

        if not(filename[-2:]!=".h5"):
            filename = filename + ".h5"
            
        with h5py.File(filename, 'w') as f:
            f.create_dataset('energy',          data=self.energy)
            f.create_dataset('glon',            data=self.glon)
            f.create_dataset('glat',            data=self.glat)
            f.create_dataset('event_times',     data=self.event_times)
            f.create_dataset('pointing_dirs',   data=self.pointing_dirs)
            f.create_dataset('_source_ids',     data=self._source_ids)
            f.create_dataset('energy_axis',     data=self.energy_axis)
            f.create_dataset('glongitude_axis', data=self.glongitude_axis)
            f.create_dataset('glatitude_axis',  data=self.glatitude_axis)


            f.attrs['obs_id']                       = self.obs_id
            f.attrs['obs_start_time']               = self.obs_start_time
            f.attrs['obs_end_time']                 = self.obs_end_time
            f.attrs['hemisphere']                   = self.hemisphere
            f.attrs['zenith_angle']                 = self.zenith_angle
            f.attrs['_likelihood_id']               = self._likelihood_id
            f.attrs['_true_vals']                   = self._true_vals

    # Class method to load an EventData object from an HDF5 file
    @classmethod
    def load_from_hdf5(cls, filename):
        with h5py.File(filename, 'r') as f:
            # Access the datasets
            energy          = np.array(f['energy'])
            glon            = np.array(f['glon'])
            glat            = np.array(f['glat'])
            event_times     = np.array(f['event_times'])
            pointing_dirs   = np.array(f['pointing_dirs'])
            _source_ids     = np.array(f['_source_ids'])
            energy_axis     = np.array(f['energy_axis'])
            glongitude_axis = np.array(f['glongitude_axis'])
            glatitude_axis  = np.array(f['glatitude_axis'])

            # Access the attributes (metadata)
            obs_id          = f.attrs['obs_id']
            obs_start_time  = f.attrs['obs_start_time']
            obs_end_time    = f.attrs['obs_end_time']
            hemisphere      = f.attrs['hemisphere']
            zenith_angle    = f.attrs['zenith_angle']
            _likelihood_id  = f.attrs['_likelihood_id']
            _true_vals      = f.attrs['_true_vals']

        return EventData(energy=energy, glon=glon, glat=glat, pointing_dirs=pointing_dirs,
                         energy_axis=energy_axis, glongitude_axis=glongitude_axis, glatitude_axis=glatitude_axis,
                         obs_id=obs_id, hemisphere=hemisphere, zenith_angle=zenith_angle,
                         event_times=event_times,
                         obs_start_time=obs_start_time, obs_end_time=obs_end_time,
                         _source_ids=_source_ids, _likelihood_id=_likelihood_id, _true_vals=_true_vals)



