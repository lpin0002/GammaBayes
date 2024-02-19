from gammabayes.likelihoods.core import DiscreteLogLikelihood
from gammabayes.likelihoods.irfs.irf_extractor_class import IRFExtractor
from gammabayes.likelihoods.irfs.irf_normalisation_setup import irf_norm_setup
import numpy as np
class IRF_LogLikelihood(DiscreteLogLikelihood):

    def __init__(self,pointing_direction=[0,0], zenith=20, hemisphere='South', prod_vers=5, *args, **kwargs):
        """_summary_

        Args:
            name (list[str] | tuple[str], optional): Identifier name(s) for the likelihood instance.

            pointing_direction (list, optional): Pointing direction of the telescope in galactic 
            coordinates (e.g. Directly pointing at the Galactic Centre is the default). Defaults to [0,0].

            zenith (int, optional): Zenith angle of the telescope (can be 20, 40 or 60 degrees). 
            Defaults to 20.

            hemisphere (str, optional): Which hemisphere the telescope observation was in, can be 'South' 
            or 'North'. Defaults to 'South'.

            prod_vers (int, optional): Version of the likelihood function, can currently be 3/3b or 5. 
            Defaults to 5.
            
            axes (list[np.ndarray] | tuple[np.ndarray]): Arrays defining the reconstructed observation value axes.
            
            dependent_axes (list[np.ndarray]): Arrays defining the true observation value axes.
            
            inputunit (str | list[str] | tuple[str], optional): Unit(s) of the input axes.
            
            axes_names (list[str] | tuple[str], optional): Names of the independent variable axes.
            
            dependent_axes_names (list[str] | tuple[str], optional): Names of the dependent variable axes.
            
            iterative_logspace_integrator (callable, optional): Integration method used for normalization.
            
            parameters (dict | ParameterSet, optional): Parameters for the log likelihood function.
        """
        self.irf_loglikelihood = IRFExtractor(zenith_angle=zenith, hemisphere=hemisphere, prod_vers=prod_vers)
        super().__init__(
            logfunction=self.irf_loglikelihood.single_loglikelihood, 
            *args, **kwargs
        )
        self.pointing_direction = pointing_direction

    def __call__(self, *args, **kwargs):
        """_summary_

        Args:
            recon_energy (float): Measured energy value by the CTA
            recon_lon (float): Measured FOV longitude of a gamma-ray event
                detected by the CTA
            recon_lat (float): Measured FOV latitude of a gamma-ray event
                detected by the CTA
            true_energy (float): True energy of a gamma-ray event detected by the CTA
            true_lon (float): True FOV longitude of a gamma-ray event 
                detected by the CTA
            true_lat (float): True FOV latitude of a gamma-ray event 
                detected by the CTA

        Returns:
            float: natural log of the full CTA likelihood for the given gamma-ray 
                event data
        """
        
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


    
