import numpy as np
import time, random, warnings, os


from gammabayes.dark_matter.channel_spectra import (
    single_channel_spectral_data_path,
    PPPCReader, 
)
from gammabayes.dark_matter.density_profiles import Einasto_Profile
from gammabayes.priors import DiscreteLogPrior

# $$
from gammabayes.likelihoods.irfs import IRF_LogLikelihood
from gammabayes.utils.config_utils import (
    read_config_file, 
    create_true_axes_from_config, 
    create_recon_axes_from_config, 
)
from gammabayes.utils import (
    dynamic_import,
)
from gammabayes.hyper_inference import MTree

from gammabayes import (
    Parameter, ParameterSet, ParameterSetCollection,
    apply_dirichlet_stick_breaking_direct,
    update_with_defaults
)
from matplotlib import pyplot as plt
from gammabayes.dark_matter import CustomDMRatiosModel, CombineDMComps


from dynesty.pool import Pool as DyPool


class hl_setup_from_config:

    def __init__(self, config):


        if isinstance(config, dict):
            self.config_dict = config
        else:
            self.config_dict = read_config_file(config)

        self.prior_parameter_specifications = self.config_dict['prior_parameter_specifications']
        self.shared_parameter_specifications = self.config_dict['shared_parameter_specifications']
        self.mixture_parameter_specifications = self.config_dict['mixture_parameter_specifications']
        

        self.save_path = self.config_dict['save_path']


        self.true_axes       = create_true_axes_from_config(self.config_dict)
        self.recon_axes      = create_recon_axes_from_config(self.config_dict)

        self._setup_mixture_tree_values()

        self._setup_irfs()
        self._handle_missing_true_specifications()
        self._handle_dm_specifications()
        self._setup_observational_prior_models()
        self._setup_parameter_specifications()
        



        
    def _setup_irfs(self):

        if 'IRF_kwarg_Specifications' in self.config_dict:
            IRF_kwarg_Specifications = self.config_dict['IRF_kwarg_Specifications']


        self.irf_loglike = IRF_LogLikelihood(axes   =   self.recon_axes, 
                                dependent_axes =   self.true_axes,
                                **IRF_kwarg_Specifications)
        


        if 'log_irf_norm_matrix_path' in self.config_dict:
            self.log_irf_norm_matrix_path = self.config_dict['log_irf_norm_matrix_path']

            try:
                print("Loaded from specified path")
                self.log_irf_norm_matrix = np.load(self.log_irf_norm_matrix_path)

            except:
                self.log_irf_norm_matrix = None
        else:
            self.log_irf_norm_matrix = None


        if ('log_edisp_norm_matrix_path' in self.config_dict) and ('log_psf_norm_matrix_path' in self.config_dict):
            
            self.log_psf_norm_matrix_load_path = self.config_dict['log_psf_norm_matrix_path']
            self.log_psf_norm_matrix = np.load(self.log_psf_norm_matrix_load_path)

            self.log_edisp_norm_matrix_path = self.config_dict['log_edisp_norm_matrix_path']
            self.log_edisp_norm_matrix = np.load(self.log_edisp_norm_matrix_path)
        else:

            self.log_psf_norm_matrix = None
            self.log_edisp_norm_matrix = None



        if self.log_irf_norm_matrix is None:
            if self.log_psf_norm_matrix is None:
                self.log_psf_norm_matrix, self.log_edisp_norm_matrix = self.irf_loglike.create_log_norm_matrices()

            self.log_irf_norm_matrix = self.log_psf_norm_matrix + self.log_edisp_norm_matrix


        
        
        if 'log_irf_norm_matrix_path' in self.config_dict:
            np.save(self.config_dict['log_irf_norm_matrix_path'], self.log_irf_norm_matrix)
        else:
            if self.save_path=="":
                self.log_irf_norm_matrix_path = 'irf_norm_matrix.npy'
            else:
                self.log_irf_norm_matrix_path = self.save_path+'/irf_norm_matrix.npy'


            np.save(self.log_irf_norm_matrix_path, self.log_irf_norm_matrix)
            self.log_psf_norm_matrix_path = os.path.abspath(self.log_irf_norm_matrix_path)

            print(f"Saving to {self.log_irf_norm_matrix_path}")

        
        if 'log_psf_norm_matrix_path' in self.config_dict:
            np.save(self.config_dict['log_psf_norm_matrix_path'], self.log_psf_norm_matrix)
        else:
            if self.save_path=="":
                self.log_psf_norm_matrix_path = 'log_psf_norm_matrix.npy'
            else:
                self.log_psf_norm_matrix_path = self.save_path+'/log_psf_norm_matrix.npy'


            np.save(self.log_psf_norm_matrix_path, self.log_psf_norm_matrix)
            self.log_psf_norm_matrix_path = os.path.abspath(self.log_psf_norm_matrix_path)
            

        

        if 'log_edisp_norm_matrix_path' in self.config_dict:
            np.save(self.config_dict['log_edisp_norm_matrix_path'], self.log_edisp_norm_matrix)
        else:
            if self.save_path=="":
                self.log_edisp_norm_matrix_path = 'log_edisp_norm_matrix.npy'
            else:
                self.log_edisp_norm_matrix_path = self.save_path+'/log_edisp_norm_matrix.npy'

                
            np.save(self.log_edisp_norm_matrix_path, self.log_edisp_norm_matrix)
            self.log_psf_norm_matrix_path = os.path.abspath(self.log_edisp_norm_matrix_path)



    def _handle_dm_specifications(self):

        if 'dark_matter_density_profile' in self.config_dict:
            density_profile_string = self.config_dict['dark_matter_density_profile']
            self.dark_matter_density_profile = dynamic_import(
                'gammabayes.dark_matter.density_profiles',  density_profile_string)
        else:
            density_profile_string = 'Einasto_Profile'
            self.dark_matter_density_profile = Einasto_Profile



        if 'dark_matter_mass' in self.config_dict:
            self.dark_matter_mass = self.config_dict['dark_matter_mass']
        else:
            self.dark_matter_mass = np.nan



        if 'dark_matter_spectral_model' in self.config_dict:
            self.use_dm_model = True
            self.use_dm_channels = False
            dm_spectral_model_string = self.config_dict['dark_matter_spectral_model']

            self.dark_matter_spectral_class = dynamic_import(
                'gammabayes.dark_matter.spectral_models',  dm_spectral_model_string)
            

            self.dark_matter_model = CombineDMComps(
                name=f'DarkMatter_{dm_spectral_model_string}_{density_profile_string}',

                spectral_class = self.dark_matter_spectral_class,
                spatial_class = self.dark_matter_density_profile,

                axes = self.true_axes,

                irf_loglike = self.irf_loglike, 

                default_spectral_parameters={'mass':self.dark_matter_mass}
                )
            

        elif 'DM_Channels' in self.config_dict:
            self.use_dm_model = False
            self.use_dm_channels = True

            self.DM_Channels = self.config_dict['DM_Channels']

            if 'DM_Annihilation_Ratios' in self.config_dict:
                self.DM_Annihilation_Ratios = self.config_dict['DM_Annihilation_Ratios']
            else:
                self.DM_Annihilation_Ratios = None


            self.custom_ratios_model = CustomDMRatiosModel(
                name='DM',
                channels=self.DM_Channels,
                irf_loglike=self.irf_loglike, 
                spatial_class=self.dark_matter_density_profile,
                axes=self.true_axes, 
                default_spectral_parameters={'mass':self.dark_matter_mass})

        else:
            self.use_dm_model = False
            self.use_dm_channels = False


        if self.use_dm_channels:
            if not('DM' in self.mixture_parameter_specifications):
                warnings.warn(
                    """When using dark matter models by channel, there must be a specifications for the
                    overall signal fraction named 'DM' within the mixture parameter specifications.""", 
                    category=UserWarning)




    def _handle_missing_true_specifications(self):

        if 'true_mixture_fractions' in self.config_dict:
            self.true_mixture_fractions = self.config_dict['prior_parameter_specifications']
        else:
            self.true_mixture_fractions = None


        if 'Nevents_per_job' in self.config_dict:
            self.Nevents_per_job = self.config_dict['Nevents_per_job']
        elif 'Nevents' in self.config_dict:
            self.Nevents_per_job = self.config_dict['Nevents']
        else:
            self.Nevents_per_job = np.nan


        if 'DM_Annihilation_Ratios' in self.config_dict:
            self.DM_Annihilation_Ratios = self.config_dict['DM_Annihilation_Ratios']
        else:
            self.DM_Annihilation_Ratios = None




    def _setup_observational_prior_models(self):
        self.observational_prior_model_names = self.config_dict['observational_prior_models']

        self.observational_prior_models = ['Not Set']*len(self.observational_prior_model_names)
        nested_model_idx_offset = 0


        for model_idx, model_name in enumerate(self.observational_prior_model_names):
            
            if not(model_name in ['DM', 'CCR_BKG']):
                prior_model_class = dynamic_import('gammabayes.priors', model_name)
                prior_model = prior_model_class(
                    energy_axis=self.true_axes[0], 
                    longitudeaxis=self.true_axes[1], 
                    latitudeaxis=self.true_axes[2], 
                    irf=self.irf_loglike)
                
                self.observational_prior_models[model_idx+nested_model_idx_offset] = prior_model

        
            elif model_name=='DM':
                if self.use_dm_model:
                    prior_model = self.dark_matter_model
                    self.observational_prior_models[model_idx+nested_model_idx_offset] = prior_model
                elif self.use_dm_channels:
                    num_channels  = len(self.custom_ratios_model.channels)
                    self.observational_prior_models.extend(['not set channels']*(num_channels-1))
                    self.observational_prior_models[model_idx:model_idx+num_channels] = list(self.custom_ratios_model.values())
                    nested_model_idx_offset+=num_channels-1



            elif model_name=='CCR_BKG':
                ccr_bkg_prior = DiscreteLogPrior(logfunction=self.irf_loglike.log_bkg_CCR, name='CCR_BKG',
                            axes=self.true_axes, 
                            axes_names=['energy', 'lon', 'lat'], )
                self.observational_prior_models[model_idx+nested_model_idx_offset] = ccr_bkg_prior




    def _setup_parameter_specifications(self):

        num_obs_priors = len(self.observational_prior_models)

        formatted_prior_parameter_specifications = {}


        for model_idx, obs_prior in enumerate(self.observational_prior_models):

            if obs_prior.name in self.prior_parameter_specifications:
                unique_parameter_spec = ParameterSet(self.prior_parameter_specifications[obs_prior.name])
            else:
                unique_parameter_spec = ParameterSet()

            formatted_prior_parameter_specifications[obs_prior.name] = unique_parameter_spec


        for parameter_name, [shared_models, parameter_spec] in self.shared_parameter_specifications.items():
            if not('name' in parameter_spec):
                parameter_spec['name'] = parameter_name


            for shared_model_name in shared_models:
                
                parameter_spec['prior_id'] = shared_model_name


                formatted_prior_parameter_specifications[shared_model_name].append(Parameter(parameter_spec))


        self.obs_prior_parameter_specifications = list(formatted_prior_parameter_specifications.values())







    def _setup_mixture_tree_values(self, values=None):

        self.mixture_layout = self.config_dict['mixture_layout']

        if values is None:

            unformatted_values = []

            if 'true_mixture_fractions' in self.config_dict:
                self.true_mixture_fractions = self.config_dict['true_mixture_fractions']
            else:
                self.true_mixture_fractions = {}



            if 'DM_Annihilation_Ratios' in self.config_dict:
                self.DM_Annihilation_Ratios = self.config_dict['DM_Annihilation_Ratios']
            else:
                self.DM_Annihilation_Ratios = {}

            print(self.true_mixture_fractions)
            print(self.DM_Annihilation_Ratios)

            values = []
            for source_name, fraction in self.true_mixture_fractions.items():


                values.append(fraction)

                if source_name=='DM':
                    values.extend(list(self.DM_Annihilation_Ratios.values()))

        if len(values) <1:
            values=None


        self.mixture_tree = MTree()
        self.mixture_tree.create_tree(self.mixture_layout, values=values)









        



        



        

        
        


