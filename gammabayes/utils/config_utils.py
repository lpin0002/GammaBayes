import warnings, yaml, sys, time


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