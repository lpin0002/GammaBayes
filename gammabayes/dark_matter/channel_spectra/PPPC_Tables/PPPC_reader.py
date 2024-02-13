import csv
import re
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm




class PPPCReader:
    # Handy conversion dictionary
    # This is done instead of change the AtProduction_gammas_... file
        # so that in the future one can swap in and out the file
        # and nothing else needs to change in the code. Even though
        # it isn't directly used in the class
        # (Assuming the header names are the same)
    darkSUSY_to_PPPC_converter = {
        "nuenue":"\\[Nu]e",
        "e+e-": "e",
        "numunumu":"\\[Nu]\\[Mu]",
        "mu+mu-":"mu",
        'nutaunutau':"\\[Nu]\\[Tau]",
        "tau+tau-":"\\[Tau]",
        "cc": "c",
        "tt": "t",
        "bb": "b",
        "ss":"q",
        "dd":"q",
        "uu":"q",
        "gammagamma": "\\[Gamma]",
        "W+W-": "W",
        "ZZ": "Z",
        "gg": "g",
        "HH": "h",
    }



    def __init__(self, file_name):
        self.file_name = file_name
        self.data_dict = self.read_data()


        self.num_log10x = np.sum(self['mDM'] == self['mDM'][0])
        self.num_mass = int(len(self['mDM']) / self.num_log10x)

        self.mass_axis = np.unique(self['mDM'])
        self.log10x_axis = np.unique(self['Log[10,x]'])

        self.output_shape = (self.num_mass, self.num_log10x)

    def preprocess_data(self, line):
        # Split the line by one or more whitespace characters, preserving quoted fields
        return re.split(r'\s+(?=(?:[^"]*"[^"]*")*[^"]*$)', line.strip())

    def read_data(self):
        with open(self.file_name, 'r') as file:
            # Preprocess and split the first line to get headers
            headers_line = file.readline()
            headers = self.preprocess_data(headers_line)
            # Initialize a dictionary with headers as keys and empty lists as values
            data_dict = {header: [] for header in headers}
            # Process each subsequent line in the file
            for line in file:
                if line.strip():  # Ensure the line is not just whitespace
                    values = self.preprocess_data(line)
                    for header, value in zip(headers, values):
                        # Attempt to convert each value to a float if possible
                        try:
                            value = float(value)
                        except ValueError:
                            pass  # Keep value as a string if conversion fails
                        data_dict[header].append(value)


        for header in data_dict.keys():
            data_dict[header] = np.asarray(data_dict[header])

        if 'mDM' in data_dict.keys():
            data_dict['mDM']/=1000  # Turn into TeV
            
        return data_dict

    # Dictionary-like access and other methods
    def __getitem__(self, key):
        return self.data_dict[key]

    def __setitem__(self, key, value):
        self.data_dict[key] = value

    def __delitem__(self, key):
        del self.data_dict[key]

    def __iter__(self):
        return iter(self.data_dict)

    def __len__(self):
        return len(self.data_dict)

    def keys(self):
        return self.data_dict.keys()

    def values(self):
        return self.data_dict.values()

    def items(self):
        return self.data_dict.items()
    

    # Allows plotting of all the channels
    def peek(self, cmap='viridis', label_size=None, *args, **kwargs):
        channel_keys = list(self.keys())[2:]
        num_channels = len(channel_keys)
        closest_square = int(np.ceil(np.sqrt(num_channels)))

        num_cols = closest_square 
        num_rows = int(np.ceil(num_channels / closest_square))

        num_log10x = np.sum(self['mDM'] == self['mDM'][0])
        num_mass = int(len(self['mDM']) / num_log10x)

        # Determine global min and max values for consistent color scaling
            # Not including 0's in minimum coz of LogNorm colorbar
        global_min = np.min([np.min(self[key][self[key]!=0]) for key in channel_keys])
        global_max = np.max([np.max(self[key]) for key in channel_keys])

        fig, axs = plt.subplots(num_rows, num_cols,  *args, sharex=True, sharey=True, **kwargs)
        axs = axs.ravel()

        for channel_idx, channel in enumerate(channel_keys):
            pcm = axs[channel_idx].pcolormesh(
                self['mDM'].reshape((num_mass, num_log10x)), 
                self['Log[10,x]'].reshape((num_mass, num_log10x)),
                self[channel].reshape((num_mass, num_log10x)),
                norm=LogNorm(vmin=global_min, vmax=global_max),  # Use global min/max for color normalization
                cmap=cmap  # Example colormap, choose as needed
            )
            axs[channel_idx].set_xscale('log')
            axs[channel_idx].set_title(channel)

        # Hide unused axes in the main grid
        for ax_idx in range(num_channels, num_rows * num_cols):
            axs[ax_idx].axis('off')

        cbar_ax = fig.add_axes([1.02, 0.1, 0.02, 0.8])  # Add colorbar axes to the right of the figure

        fig.colorbar(pcm, cax=cbar_ax).set_label(label=r'$dN/d(log_{10}(x))$', size=label_size)

        # Add common xlabel and ylabel
        fig.text(0.5, 0.04, 'mDM [TeV]', ha='center', va='center', fontsize=label_size)
        fig.text(0.04, 0.5, 'x=mDM/Energy', ha='center', va='center', rotation='vertical', fontsize=label_size)

        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust the layout

        return fig, axs 

