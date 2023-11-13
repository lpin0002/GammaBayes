import warnings, yaml, sys, time, pickle
from gammabayes.utils.event_axes import create_axes
import numpy as np

def read_config_file(file_path):
    print(f"file path: {file_path}")
    try:
        with open(file_path, 'r') as file:
            inputs = yaml.safe_load(file)
        return inputs
    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError:
        print(f"Error: Unable to parse YAML in '{file_path}'. Please ensure it is valid JSON.")
        sys.exit(1)
        
        
def check_necessary_config_inputs(input_dict):
    
    print('\n\n')

    try:
        print(f"Number of events for this script is {input_dict['Nevents']}")
    except:
        raise Exception("Number of events to be simulated/analysed not provided. Add line 'Nevents:  [number value]' to yaml file")
            
    try:
        print(f"Provided identifier is {input_dict['identifier']}")
    except:
        warnings.warn("Identifier not provided. Default value of date in format [yr]_[month]_[day]_[hour]_[minute] will be used.", UserWarning)
        input_dict['identifier'] = time.strftime("%y_%m_%d_%H_%M")
        
    try:
        print(f"Run number is {input_dict['runnumber']}")
    except:
        input_dict['runnumber'] = 1
        
    try:
        print(f"Total number of events is {input_dict['totalevents']}")
    except:
        input_dict['totalevents'] = input_dict['Nevents']
        
    print('\n\n')


def load_hyperparameter_pickle(file_path):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def add_event_axes_config(config_dict, energy_axis_true, longitudeaxistrue, latitudeaxistrue, 
    energy_axis, longitudeaxis, latitudeaxis):
    """Takes in config dict and relevant event axes and saves min, max, spacing/resolution
        for each.

    Args:
        config_dict (dict): dict containing run information
        energy_axis_true (array_like):    energy_axis_true for analysis
        longitudeaxistrue (array_like): longitudeaxistrue for analysis
        latitudeaxistrue (array_like):  latitudeaxistrue for analysis
        energy_axis (array_like):        energy_axis for analysis
        longitudeaxis (array_like):     longitudeaxis for analysis
        latitudeaxis (array_like):      latitudeaxis for analysis

    Returns:
        dict: config file with added information
    """
    config_dict['true_energy_min']                = energy_axis_true.min()
    config_dict['true_energy_max']                = energy_axis_true.max()
    config_dict['true_energy_bins_per_decade']    = (len(energy_axis_true)-1)/np.log10(energy_axis_true).ptp()

    config_dict['true_spatial_res']               = np.diff(longitudeaxistrue)[0]
    config_dict['true_longitude_min']             = longitudeaxistrue.min()
    config_dict['true_longitude_max']             = longitudeaxistrue.max()
    config_dict['true_latitude_min']              = latitudeaxistrue.min()
    config_dict['true_latitude_max']              = latitudeaxistrue.max()


    config_dict['recon_energy_min']               = energy_axis.min()
    config_dict['recon_energy_max']               = energy_axis.max()
    config_dict['recon_energy_bins_per_decade']   = (len(energy_axis)-1)/np.log10(energy_axis).ptp()

    config_dict['recon_spatial_res']              = np.diff(longitudeaxistrue)[0]
    config_dict['recon_longitude_min']            = longitudeaxis.min()
    config_dict['recon_longitude_max']            = longitudeaxis.max()
    config_dict['recon_latitude_min']             = latitudeaxis.min()
    config_dict['recon_latitude_max']             = latitudeaxis.max()

    return config_dict




def save_config_file(config_dict, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)
    print("YAML saved to config_dict")



def create_true_axes_from_config(config_dict):

    return create_axes(config_dict['true_energy_min'], config_dict['true_energy_max'], 
                     config_dict['true_energy_bins_per_decade'], config_dict['true_spatial_res'], 
                     config_dict['true_longitude_min'], config_dict['true_longitude_max'],
                     config_dict['true_latitude_min'], config_dict['true_latitude_max'])


def create_recon_axes_from_config(config_dict):
    return create_axes(config_dict['recon_energy_min'], config_dict['recon_energy_max'], 
                     config_dict['recon_energy_bins_per_decade'], config_dict['recon_spatial_res'], 
                     config_dict['recon_longitude_min'], config_dict['recon_longitude_max'],
                     config_dict['recon_latitude_min'], config_dict['recon_latitude_max'])