import numpy as np
from matplotlib import pyplot as plt, colors
from gammabayes.utils.plotting import bin_centres_to_edges

class EventData(object):

    def __init__(self, energy=None, glon=None, glat=None, data=None, 
                 event_times=None, pointing_dirs=None, 
                 obs_start_time=None, obs_end_time=None,
                 hemisphere=None, zenith_angle=None, obs_id='No identifier',
                 recon_energy_axis=None, recon_glongitude_axis=None, recon_glatitude_axis=None):
        
        if not(energy is None):
            self.energy             = np.asarray(energy)
            self.glon               = np.asarray(glon)
            self.glat               = np.asarray(glat)

            self.data = [*zip(self.energy, self.glon, self.glat)]


            if not(event_times is None):
                self.event_times = np.asarray(event_times)

            else:
                self.event_times = np.full(shape=len(self.energy), fill_value=None)

            if not(pointing_dirs is None):
                self.pointing_dirs = np.asarray(pointing_dirs)

            else:
                self.pointing_dirs = np.full(shape=(len(self.energy), 2) , fill_value=[0.,0.], )


        
        elif not(data is None):
            self.data = np.asarray(data)

            self.energy             = np.asarray(self.data[:,0])
            self.glon               = np.asarray(self.data[:,1])
            self.glat               = np.asarray(self.data[:,2])


            if self.data.shape[1]>3:
                self.pointing_dirs  = np.asarray(self.data[:,3])
                self.event_times    = np.asarray(self.data[:,4])

            else:
                if not(event_times is None):
                    self.event_times = np.asarray(event_times)

                else:
                    self.event_times = np.full(shape=len(self.energy), fill_value=None)

                if not(pointing_dirs is None):
                    self.pointing_dirs = np.asarray(pointing_dirs)

                else:
                    self.pointing_dirs = np.full(shape=(len(self.energy), 2) , fill_value=[None,None], )
            
        self.obs_id             = obs_id
        self.hemisphere         = hemisphere
        self.zenith_angle       = zenith_angle
        self.obs_start_time     = obs_start_time
        self.obs_end_time       = obs_end_time

        self.recon_energy_axis      = recon_energy_axis
        try:
            len(self.recon_energy_axis)
            self.energy_bins        = bin_centres_to_edges(recon_energy_axis)
        except:
            self.energy_bins        = np.logspace(-1.5,2.5,81)

        self.recon_glongitude_axis  = recon_glongitude_axis
        self.recon_glatitude_axis   = recon_glatitude_axis

        try:
            len(self.recon_glongitude_axis)
            self.angular_bins        = recon_glongitude_axis
        except:
            self.angular_bins        = 101


    


    @property
    def Nevents(self):
        return len(self.energy)
    
    @property
    def full_data(self):
        return [*zip(self.energy, self.glon, self.glat, self.pointing_dirs, self.event_times)]
    


    def __iter__(self):
        self._current_datum_idx = 0  # Reset the index each time iter is called
        return self

    def __next__(self):
        if self._current_datum_idx < len(self.data):
            current_data = self.data[self._current_datum_idx]
            self._current_datum_idx += 1
            return current_data
        else:
            raise StopIteration
        

    def __str__(self):
        num_spaces = 30



        strinfo = "\n"
        strinfo += "="*num_spaces+"\n"
        strinfo += f"Event ID: {self.obs_id}\n"
        strinfo += ":"*num_spaces+"\n"
        strinfo += f"Nevents:              {self.Nevents}\n"
        strinfo += f"Hemisphere:           {self.hemisphere}\n"
        strinfo += f"Zenith Angle:         {self.zenith_angle}\n"
        strinfo += "-"*num_spaces+"\n"
        strinfo += f"Time Start:           {self.obs_start_time}\n"
        strinfo += f"Time End:             {self.obs_end_time}\n"

        try:
            strinfo += f"Min Event Time:       {min(self.event_times)}\n"
        except TypeError:
            strinfo += f"Min Event Time:       NA\n"

        try:
            strinfo += f"Max Event Time:       {max(self.event_times)}\n"
        except TypeError:
            strinfo += f"Max Event Time:       NA\n"


        strinfo += "-"*num_spaces+"\n"


        strinfo += f"Min Energy:            {min(self.energy)}\n"
        strinfo += f"Max Energy:            {max(self.energy)}\n"

        strinfo += "="*num_spaces+"\n"

        return strinfo
    



    def update(self, energy=None, glon=None, glat=None, pointing_dirs=None, event_times=None, data=None):

        if not(energy is None) and (np.asarray(energy).ndim==1):

            data = np.asarray([*zip(energy, glon, glat)])

            self.energy = np.append(self.energy, np.asarray(energy))
            self.glon   = np.append(self.glon, np.asarray(glon))
            self.glat   = np.append(self.glat, np.asarray(glat))



        elif (np.asarray(energy).ndim==2):
            data=np.asarray(energy)


        self.data = np.append(self.data, np.asarray(data), axis=0)

        self.energy = self.data[:,0]
        self.glon   = self.data[:,1]
        self.glat   = self.data[:,2]



        if not(pointing_dirs is None):
            self.pointing_dirs  = np.append(self.pointing_dirs, pointing_dirs)

        elif data.shape[1]>3:
            self.pointing_dirs  = np.append(self.pointing_dirs, data[:,3])

        else:
            self.pointing_dirs  = np.append(self.pointing_dirs, np.full(shape=(len(data[:,0]),2), fill_value=[None, None]))


        if not(event_times is None):
            self.event_times  = np.append(self.event_times, event_times)

        elif data.shape[1]>3:
            self.event_times  = np.append(self.event_times, data[:,4])
            
        else:
            self.event_times  = np.append(self.event_times, np.full(shape=(len(data[:,0]),), fill_value=None))





    
    def peek(self, figsize=(10,4), colorscale='log', *args, **kwargs):
        if colorscale == 'log':
            colorscale = colors.LogNorm()
        else:
            colorscale = None

        fig, ax = plt.subplots(1,2, figsize=figsize, *args, **kwargs)
        ax[0].hist(self.energy, bins=self.energy_bins)
        ax[0].set_xlabel('Energy [TeV]')
        ax[0].set_ylabel('Num Events')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')

        hist2d = ax[1].hist2d(self.glon, self.glat, bins=self.angular_bins, norm=colorscale)
        ax[1].set_xlabel('Galactic Longitude [deg]')
        ax[1].set_ylabel('Galactic Latitude [deg]')
        plt.colorbar(hist2d[3])
        
        fig.tight_layout()

        return fig, ax




