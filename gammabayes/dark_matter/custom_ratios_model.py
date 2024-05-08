import numpy as np, copy
from os import path
ScalarSinglet_Folder_Path = path.dirname(__file__)
from collections.abc import Iterable   # import directly from collections for Python < 3.3

from gammabayes.dark_matter.spectral_models.core import DM_ContinuousEmission_Spectrum, CSVDictionary
import time

from gammabayes.dark_matter.channel_spectra import (
    single_channel_spectral_data_path,
    PPPCReader, 
)
import warnings
from gammabayes.dark_matter import CombineDMComps
from gammabayes.dark_matter.density_profiles import Einasto_Profile
from gammabayes.dark_matter.channel_spectra import (
    single_channel_spectral_data_path,
    PPPCReader, 
    SingleDMChannel
)
from gammabayes.likelihoods import DiscreteLogLikelihood


from gammabayes.dark_matter.density_profiles import DM_Profile
from gammabayes import (
    ParameterSet, EventData, 
    apply_dirichlet_stick_breaking_direct, update_with_defaults
)


class CustomDMRatiosModel(object):

    def __init__(self, 
                 irf_loglike:DiscreteLogLikelihood, 
                 axes: list | tuple | np.ndarray,
                 spatial_class: DM_Profile = Einasto_Profile, 

                 channels: list[str] | str = 'all',
                 default_spectral_parameters: dict = {},
                 default_spatial_parameters: dict = {},
                 *args, **kwargs
                 ):
        
        # Getting all possible channels as in the PPPC tables
        all_channels = list(PPPCReader.darkSUSY_to_PPPC_converter.keys())
        self.is_single_channel = False

        
        if type(channels) == str:
            if channels.lower()=='all':
                # Ensuring that the first few elements are 
                    # 'W+W-', 'ZZ', 'HH', 'tt', 'bb', 'gg' in that order
                all_channels.remove('W+W-')
                all_channels.remove('HH')
                all_channels.remove('ZZ')
                all_channels.remove('tt')
                all_channels.remove('bb')
                all_channels.remove('gg')

                all_channels.insert(0, 'W+W-')
                all_channels.insert(1, 'ZZ')
                all_channels.insert(2, 'HH')
                all_channels.insert(3, 'tt')
                all_channels.insert(4, 'bb')
                all_channels.insert(5, 'gg')

                self.channels = all_channels
            elif channels in all_channels:

                warnings.warn("If specifying a single channel we recommend the use of the `SingleDMChannel` class in the gammabayes.dark_matter.channel_spectra module.")
                self.is_single_channel = True
                self.channels = [channels]
            else:
                raise ValueError(f"Invalid channel input given must be one or multiple of the following channels: {all_channels}")
        elif type(channels) == list:
            incorrect_channels = []

            for channel in channels:
                if channel not in all_channels:
                    incorrect_channels.append(channel)

            if len(incorrect_channels):
                raise ValueError(f"Invalid channels given ->{incorrect_channels} must be one or multiple of the following channels: {all_channels}")
            
            self.channels = channels


        self.channel_prior_dict = {}


        for channel in self.channels:
            self.channel_prior_dict[channel] = CombineDMComps(name=f"{channel} DM Class",
                        spectral_class = SingleDMChannel, 
                        spatial_class = spatial_class,
                        irf_loglike=irf_loglike, 
                        axes=axes, 
                        axes_names=['energy', 'lon', 'lat'],
                        default_spectral_parameters=default_spectral_parameters,
                        default_spatial_parameters=default_spatial_parameters,
                        )
                


    def generate_parameter_specifications(self, set_specifications_to_duplicate: dict = {}) -> dict:

        input_set = set_specifications_to_duplicate

  
        all_parameter_sets = {}
        for channel in self.channels:
            all_parameter_sets[f'{channel} DM Class'] = ParameterSet(input_set)

        return all_parameter_sets



        
    def __iter__(self):
        return self.channel_prior_dict.__iter__()
    
    def items(self):
        return self.channel_prior_dict.items()
    
    def keys(self):
        return self.channel_prior_dict.keys()
    
    def values(self):
        return self.channel_prior_dict.values()
    

    def sample(self, 
               numevents: int| list = None,
               ) -> dict[str, EventData]:
        
        if type(numevents) == int:
            if self.is_single_channel:
                return self.channel_prior_dict.values()[0].sample(numevents)
            
            event_data = {}
            for channel, channel_prior in self.channel_prior_dict.items():
                event_data[channel] = channel_prior.sample(numevents)

            return event_data
        

        
        elif isinstance(numevents, Iterable):
            if isinstance(numevents, dict):
                event_data = {}
                for channel, channel_prior in self.channel_prior_dict.items():
                    event_data[channel] = channel_prior.sample(numevents[channel])

            else:
                event_data = {}
                for channel_idx, [channel, channel_prior] in enumerate(self.channel_prior_dict.items()):
                    event_data[channel] = channel_prior.sample(numevents[channel_idx])

            return event_data
        

        else:
            raise ValueError("""'numevents' must be an int or some iterably 
(preferably a dictionary with keys being the channels or list in the same order as specified channels).""")
        

    def sample_from_weights(self, 
                            num_events: int, 
                            input_weights: list[float] | dict[float], 
                            stick_breaking: bool = True, 
                            exhaustive_fractions=False) -> dict[str, EventData]:
        


        
        if type(input_weights) == dict:
            if exhaustive_fractions:
                input_weights = [input_weights[key] for key in self.channels[:-1]]
            else:
                input_weights = [input_weights[key] for key in self.channels]
        
        if stick_breaking:
            formatted_weights = []
            for depth, channel in enumerate(self.channels):
                formatted_weight = apply_dirichlet_stick_breaking_direct(input_weights, depth) 
                # print(channel, depth, formatted_weight)
                formatted_weights.append(formatted_weight)

        else:
            formatted_weights = input_weights

        return {channel: prior.sample(int(round(weight*num_events))) for weight, [channel, prior] in zip(formatted_weights, self.channel_prior_dict.items())}





        

        





