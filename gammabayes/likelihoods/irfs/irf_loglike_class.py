from gammabayes.likelihoods.core import discrete_loglike
from gammabayes.likelihoods.irfs.irf_extractor_class import irf_extractor
from gammabayes.likelihoods.irfs.irf_normalisation_setup import irf_norm_setup
import numpy as np
class irf_loglikelihood(discrete_loglike):

    def __init__(self,pointing_direction=[0,0], zenith=20, hemisphere='South', prod_vers=5, *args, **kwargs):
        self.irf_loglikelihood = irf_extractor(zenith_angle=zenith, hemisphere=hemisphere, prod_vers=prod_vers)
        super().__init__(
            logfunction=self.irf_loglikelihood.single_loglikelihood, 
            *args, **kwargs
        )
        self.pointing_direction = pointing_direction

    def __call__(self, *args, **kwargs):
        
        return self.logfunction(pointing_direction=self.pointing_direction, *args, **kwargs)
    
         
    

    def log_edisp(self, *args, **kwargs):
        return self.irf_loglikelihood.log_edisp(pointing_direction=self.pointing_direction, *args, **kwargs)
    
    def log_psf(self, *args, **kwargs):
        return self.irf_loglikelihood.log_psf(pointing_direction=self.pointing_direction, *args, **kwargs)
    
    def log_aeff(self, *args, **kwargs):
            return self.irf_loglikelihood.log_aeff(pointing_direction=self.pointing_direction, *args, **kwargs)
    

    def create_log_norm_matrices(self, **kwargs):
        return irf_norm_setup(energy_true_axis=self.dependent_axes[0],
                            longitudeaxistrue=self.dependent_axes[1],
                            latitudeaxistrue=self.dependent_axes[2],

                            energy_recon_axis=self.axes[0],
                            longitudeaxis=self.axes[1],
                            latitudeaxis=self.axes[2],
                            
                            log_psf=self.log_psf,
                            log_edisp=self.log_edisp,
                            
                            **kwargs
                            )


    
