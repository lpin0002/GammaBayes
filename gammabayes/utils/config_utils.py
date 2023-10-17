
import yaml, sys, pickle
from gammabayes.utils.utils import create_axes


def read_config_file(file_path):
    print(f"config file path: {file_path}")
    try:
        with open(file_path, 'r') as file:
            inputs = yaml.safe_load(file)
        return inputs
    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError:
        print(f"Error: Unable to parse YAML in '{file_path}'. Please ensure it is valid yaml file.")
        sys.exit(1)

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