import numpy as np, time
from gammabayes.dark_matter.spectral_models import (
    DM_ContinuousEmission_Spectrum
)
from os import path
from gammabayes.dark_matter.density_profiles import DM_Profile
from gammabayes.priors.core import TwoCompPrior
from gammabayes.utils import update_with_defaults
from gammabayes.utils import logspace_simpson


class CombineDMComps(TwoCompPrior):

    def __init__(self, 
                 spectral_class: DM_ContinuousEmission_Spectrum, 
                 spatial_class: DM_Profile, 
                 *args, **kwargs
                 ):
        

        super().__init__(spectral_class=spectral_class, 
                         spatial_class =spatial_class,
                         *args, **kwargs
        )

    def convert_param_to_logsigmav(self,
                            signal_fraction, 
                            true_axes=None,
                            spectral_parameters = {},
                            spatial_parameters = {},
                            totalnumevents=1e8, 
                            tobs_seconds=525*60*60, symmetryfactor=1):
        
        update_with_defaults(spectral_parameters, self.default_spectral_parameters)
        update_with_defaults(spatial_parameters, self.default_spatial_parameters)


        if true_axes is None:
            true_axes = self.axes
        
    
        if self.mesh_efficient_exists:
            integrand = self.log_dist_mesh_efficient(*true_axes, 
                                                     spectral_parameters=spectral_parameters,
                                                     spatial_parameters=spatial_parameters)
            

        # Splitting it up for easy debuggin
        log_integrated_energy = logspace_simpson(
                                    logy=integrand, x = true_axes[0], axis=0)
        

        
        log_integrated_energy_longitude = logspace_simpson(
                                logy=log_integrated_energy, 
                                    x = true_axes[1], axis=0)
        
                        
        logintegral = logspace_simpson(
                            logy=log_integrated_energy_longitude.T*np.cos(true_axes[2]*np.pi/180), 
                                    x = true_axes[2], axis=-1).T
        

        logsigmav = np.log(8*np.pi*symmetryfactor*spectral_parameters['mass']**2*totalnumevents*signal_fraction) - logintegral - np.log(tobs_seconds)

        print(logsigmav)
        return np.exp(logsigmav)


    def convert_param_to_logsigmav_integral_efficient(self,
                                signal_fraction, 
                                true_axes=None,
                                spectral_parameters = {},
                                spatial_parameters = {},
                                totalnumevents=1e8, 
                                tobs_seconds=525*60*60, symmetryfactor=1):
        
        update_with_defaults(spectral_parameters, self.default_spectral_parameters)
        update_with_defaults(spatial_parameters, self.default_spatial_parameters)


        if true_axes is None:
            true_axes = self.axes
        
    
        if self.mesh_efficient_exists:
            integrand = self.log_dist_integral_mesh_efficient(*true_axes, 
                                                     spectral_parameters=spectral_parameters,
                                                     spatial_parameters=spatial_parameters)
            

        # Splitting it up for easy debuggin
        log_integrated_energy = logspace_simpson(
                                    logy=integrand, x = true_axes[0], axis=0)
        

        
        log_integrated_energy_longitude = logspace_simpson(
                                logy=log_integrated_energy, 
                                    x = true_axes[1], axis=0)
        
                        
        logintegral = logspace_simpson(
                            logy=log_integrated_energy_longitude.T*np.cos(true_axes[2]*np.pi/180), 
                                    x = true_axes[2], axis=-1).T.reshape(signal_fraction.shape)
        

        logsigmav = np.log(8*np.pi*symmetryfactor*spectral_parameters['mass']**2*totalnumevents*signal_fraction) - logintegral - np.log(tobs_seconds)


        return np.exp(logsigmav)
    

