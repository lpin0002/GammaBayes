import warnings, yaml, sys, time, pickle
from ..utils.event_axes import create_axes


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
        print(f"Provided log10 mass is {input_dict['logmass']}")
    except:
        raise Exception("Log10 mass not provided. Add line 'logmass:  [logmass value]' to yaml file")
    
    try:
        print(f"Provided signal fraction/xi is {input_dict['xi']}")
    except:
        raise Exception("Signal fraction not provided. Add line 'xi:  [fractional value]' to yaml file")
    
    try:
        print(f"Provided identifier is {input_dict['identifier']}")
    except:
        warnings.warn("Identifier not provided. Default value of date in format [yr]_[month]_[day]_[hour]_[minute] will be used.", UserWarning)
        input_dict['identifier'] = time.strftime("%y_%m_%d_%H_%M")
        
    try:
        print(f"Number of mass bins to be tested is {input_dict['nbins_logmass']}")
    except:
        warnings.warn("Number of mass bins not provided. Default value of 61 to be used.", UserWarning)
        input_dict['nbins_logmass'] = 61
        
    try:
        print(f"Number of signal fraction bins to be tested is {input_dict['nbins_xi']}")
    except:
        warnings.warn("Number of signal fraction bins not provided. Default value of 101 to be used.", UserWarning)
        input_dict['nbins_xi'] = 101
        
        
    try:
        print(f"Dark matter density profile is {input_dict['dmdensity_profile']}")
    except:
        warnings.warn("Dark matter density profile not provided. Default value of 'einasto' will be used.", UserWarning)
        input_dict['dmdensity_profile'] = 'einasto'
        
    try:
        print(f"Number of cores to be used is {input_dict['numcores']}")
    except:
        input_dict['numcores'] = 1
        
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


def save_config_file(config_dict, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)
    print("Configuration saved to config_dict")



def create_true_axes_from_config(config_dict):

    return create_axes(config_dict['log10_true_energy_min'], config_dict['log10_true_energy_max'], 
                     config_dict['log10_true_energy_bins_per_decade'], config_dict['true_spatial_res'], 
                     config_dict['true_longitude_min'], config_dict['true_longitude_max'],
                     config_dict['true_latitude_min'], config_dict['true_latitude_max'])


def create_recon_axes_from_config(config_dict):
    return create_axes(config_dict['log10_recon_energy_min'], config_dict['log10_recon_energy_max'], 
                     config_dict['log10_recon_energy_bins_per_decade'], config_dict['recon_spatial_res'], 
                     config_dict['recon_longitude_min'], config_dict['recon_longitude_max'],
                     config_dict['recon_latitude_min'], config_dict['recon_latitude_max'])