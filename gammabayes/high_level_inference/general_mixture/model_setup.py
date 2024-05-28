import numpy as np, time, random


from gammabayes.dark_matter.channel_spectra import (
    single_channel_spectral_data_path,
    PPPCReader, 
)
from gammabayes.dark_matter.density_profiles import Einasto_Profile


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
from gammabayes import (
    Parameter, ParameterSet, ParameterSetCollection,
    apply_dirichlet_stick_breaking_direct,
    update_with_defaults
)
from matplotlib import pyplot as plt
from gammabayes.dark_matter import CustomDMRatiosModel, CombineDMComps


from dynesty.pool import Pool as DyPool


class model_setup_from_config:

    def __init__(self, config_filename):

        self.config_dict = read_config_file(config_filename)

        self.prior_parameter_specifications = self.config_dict['prior_parameter_specifications']
        self.shared_parameter_specifications = self.config_dict['shared_parameter_specifications']
        self.mixture_parameter_specifications = self.config_dict['mixture_parameter_specifications']
        
        self.save_path = self.config_dict['save_path']

        self.true_axes       = create_true_axes_from_config(self.config_dict)
        self.recon_axes      = create_recon_axes_from_config(self.config_dict)
        self._setup_irfs()



        
    def _setup_irfs(self):

        if 'IRF_kwarg_Specifications' in self.config_dict:
            IRF_kwarg_Specifications = self.config_dict['IRF_kwarg_Specifications']


        self.irf_loglike = IRF_LogLikelihood(axes   =   self.recon_axes, 
                                dependent_axes =   self.true_axes,
                                **IRF_kwarg_Specifications)
        


        if 'log_irf_norm_matrix_load_path' in self.config_dict:
            self.log_irf_norm_matrix_path = self.config_dict['log_irf_norm_matrix_load_path']
            try:
                self.log_irf_norm_matrix = np.load(self.log_irf_norm_matrix_path)
            except:
                self.log_irf_norm_matrix = None
        else:
            self.log_irf_norm_matrix = None


        if ('log_psf_norm_matrix_load_path' in self.config_dict) and ('log_edisp_norm_matrix_load_path' in self.config_dict):
            self.log_psf_norm_matrix_path = self.config_dict['log_psf_norm_matrix_load_path']
            self.log_psf_norm_matrix = np.load(self.log_psf_norm_matrix_path)

            self.log_edisp_norm_matrix_path = self.config_dict['log_edisp_norm_matrix_load_path']
            self.log_edisp_norm_matrix = np.load(self.log_edisp_norm_matrix_path)
        else:
            self.log_psf_norm_matrix = None
            self.log_edisp_norm_matrix = None


        if self.log_irf_norm_matrix is None:
            if self.log_psf_norm_matrix is None:
                self.log_psf_norm_matrix, self.log_edisp_norm_matrix = self.irf_loglike.create_log_norm_matrices()

            self.log_irf_norm_matrix = self.log_psf_norm_matrix + self.log_edisp_norm_matrix


        
        
        if 'log_irf_norm_matrix_save_path' in self.config_dict:
            np.save(self.config_dict['log_irf_norm_matrix_save_path'], self.log_irf_norm_matrix)
        else:
            np.save(self.save_path+'irf_norm_matrix.npy', self.log_irf_norm_matrix)

        
        if 'log_psf_matrix_save_path' in self.config_dict:
            np.save(self.config_dict['log_psf_matrix_save_path'], self.log_psf_norm_matrix)
        else:
            np.save(self.save_path+'log_psf_norm_matrix.npy', self.log_psf_norm_matrix)
        

        if 'log_edisp_matrix_save_path' in self.config_dict:
            np.save(self.config_dict['log_edisp_matrix_save_path'], self.log_edisp_norm_matrix)
        else:
            np.save(self.save_path+'log_edisp_norm_matrix.npy', self.log_edisp_norm_matrix)




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
            



        elif 'DM_Channels' in self:
            self.use_dm_model = False
            self.use_dm_channels = True

            self.DM_Channels = self.config_dict['DM_Channels']

            if 'DM_Annihilation_Ratios' in self.config_dict:
                self.DM_Annihilation_Ratios = self.config_dict['DM_Annihilation_Ratios']
            else:
                self.DM_Annihilation_Ratios = None


            self.custom_ratios_model = CustomDMRatiosModel(
                channels=self.DM_Channels,
                irf_loglike=self.irf_loglike, 
                spatial_class=self.dark_matter_density_profile,
                axes=self.true_axes, 
                default_spectral_parameters={'mass':self.dark_matter_mass})

        else:
            self.use_dm_model = False
            self.use_dm_channels = False




    def _handle_missing_true_specifications(self):

        if 'true_mixture_fractions' in self.config_dict:
            self.true_mixture_fractions = self.config_dict['prior_parameter_specifications']
        else:
            self.true_mixture_fractions = None


        if 'Nevents_per_job' in self.config_dict:
            self.Nevents_per_job = self.config_dict['Nevents_per_job']
        else:
            self.Nevents_per_job = np.nan



        



        

        
        


