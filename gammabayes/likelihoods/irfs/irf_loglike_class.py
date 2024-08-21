from gammabayes.likelihoods.core import DiscreteLogLikelihood
from gammabayes.likelihoods.irfs.irf_extractor_class import IRFExtractor
from gammabayes.likelihoods.irfs.irf_normalisation_setup import irf_norm_setup
from astropy import units as u
from astropy.units import Quantity
import numpy as np
class IRF_LogLikelihood(DiscreteLogLikelihood):

    def __init__(self,pointing_dir:list[Quantity]=[0*u.deg,0*u.deg], 
                 zenith:int=20, hemisphere:str='South', 
                 observation_time: float|u.Quantity = 50*u.hr,
                 prod_vers=5, *args, **kwargs):
        """_summary_

        Args:
            name (list[str] | tuple[str], optional): Identifier name(s) for the likelihood instance.

            pointing_dir (list, optional): Pointing direction of the telescope in galactic 
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
        self.irf_loglikelihood = IRFExtractor(zenith_angle=zenith, hemisphere=hemisphere, prod_vers=prod_vers, observation_time=observation_time)
        super().__init__(
            logfunction=self.irf_loglikelihood.single_loglikelihood, 
            *args, **kwargs
        )
        self.pointing_dir = pointing_dir


        self.psf_units = self.irf_loglikelihood.psf_units
        self.edisp_units = self.irf_loglikelihood.edisp_units
        self.aeff_units = self.irf_loglikelihood.aeff_units
        self.CCR_BKG_units = self.irf_loglikelihood.CCR_BKG_units


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
        
        return self.logfunction(pointing_dir=self.pointing_dir, *args, **kwargs)
    

    def log_edisp(self, *args, **kwargs):
        """
        Wrapper for the Gammapy interpretation of the CTA energy dispersion function.

        Args:
            recon_energy (Quantity): Measured energy value by the CTA.
            true_energy (Quantity): True energy of a gamma-ray event detected by the CTA.
            true_lon (Quantity): True FOV longitude of a gamma-ray event detected by the CTA.
            true_lat (Quantity): True FOV latitude of a gamma-ray event detected by the CTA.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: Natural log of the CTA energy dispersion likelihood for the given gamma-ray event data.
        """
        return self.irf_loglikelihood.log_edisp(pointing_dir=self.pointing_dir, *args, **kwargs)
    
    def log_psf(self, *args, **kwargs):
        """
        Wrapper for the Gammapy interpretation of the CTA point spread function.

        Args:
            recon_lon (Quantity): Measured FOV longitude of a gamma-ray event detected by the CTA.
            recon_lat (Quantity): Measured FOV latitude of a gamma-ray event detected by the CTA.
            true_energy (Quantity): True energy of a gamma-ray event detected by the CTA.
            true_lon (Quantity): True FOV longitude of a gamma-ray event detected by the CTA.
            true_lat (Quantity): True FOV latitude of a gamma-ray event detected by the CTA.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: Natural log of the CTA point spread function likelihood for the given gamma-ray event data.
        """
        return self.irf_loglikelihood.log_psf(pointing_dir=self.pointing_dir, *args, **kwargs)
    
    def log_aeff(self, *args, pointing_dir =None, **kwargs):
        """
        Wrapper for the Gammapy interpretation of the log of the CTA effective area function.

        Args:
            energy (Quantity): True energy of a gamma-ray event detected by the CTA.
            longitude (Quantity): True FOV longitude of a gamma-ray event detected by the CTA.
            latitude (Quantity): True FOV latitude of a gamma-ray event detected by the CTA.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: The natural log of the effective area of the CTA in cm^2.
        """
        if pointing_dir is None:
            pointing_dir = self.pointing_dir

        return self.irf_loglikelihood.log_aeff(pointing_dir=self.pointing_dir, *args, **kwargs)
    

    def log_bkg_CCR(self, *args, pointing_dir=None, **kwargs):
        """
        Wrapper for the Gammapy interpretation of the log of the CTA's background charged cosmic-ray mis-identification rate.

        Args:
            energy (Quantity): True energy of a gamma-ray event detected by the CTA.
            longitude (Quantity): True FOV longitude of a gamma-ray event detected by the CTA.
            latitude (Quantity): True FOV latitude of a gamma-ray event detected by the CTA.
            spectral_parameters (dict, optional): Spectral parameters. Defaults to {}.
            spatial_parameters (dict, optional): Spatial parameters. Defaults to {}.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: Natural log of the charged cosmic ray mis-identification rate for the CTA.
        """
        if pointing_dir is None:
            pointing_dir = self.pointing_dir

            
        return self.irf_loglikelihood.log_bkg_CCR(pointing_dir=pointing_dir, *args, **kwargs)


    def create_log_norm_matrices(self, **kwargs):
        """
        Creates normalization matrices for the IRF log likelihood.

        Args:
            **kwargs: Additional parameters for the normalization setup.

        Returns:
            dict: Normalization matrices for the log likelihood.
        """
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
    
    def to_dict(self):
        data_dict = {}
        data_dict["pointing_dir"] = self.pointing_dir
        data_dict["zenith"] = self.zenith
        data_dict["hemisphere"] = self.hemisphere
        data_dict["prod_vers"] = self.prod_vers
        data_dict["observation_time"] = self.observation_time

        return data_dict



    
