from gammabayes.utils import convertlonlat_to_offset, angularseparation, resources_dir
import numpy as np
from astropy import units as u
from gammapy.irf import load_irf_dict_from_file
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom


class irf_extractor(object):
    def __init__(self, file_path=resources_dir+'/irf_fits_files/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits'):
        self.file_path = file_path

    
        extracted_default_irfs  = load_irf_dict_from_file(self.file_path)

        self.edisp_default      = extracted_default_irfs['edisp']
        self.psf_default        = extracted_default_irfs['psf']

        self.psf3d              = self.psf_default.to_psf3d()
        self.aeff_default       = extracted_default_irfs['aeff']

        self.aefffunc = lambda energy, offset: self.aeff_default.evaluate(energy_true = energy*u.TeV, offset=offset*u.deg).to(u.cm**2).value

    def log_aeff(self, true_energy, true_lon, true_lat, pointing_direction=None):
        """Wrapper for the Gammapy interpretation of the log of 
            the CTA effective area function.

        Args:
            true_energy (float): True energy of a gamma-ray event detected by the CTA
            true_lon (float): True FOV longitude of a gamma-ray event 
                detected by the CTA
            true_lat (float): True FOV latitude of a gamma-ray event 
                detected by the CTA

        Returns:
            float: The natural log of the effective area of the CTA in m^2
        """
        return np.log(self.aeff_default.evaluate(energy_true = true_energy*u.TeV, 
                                offset=convertlonlat_to_offset(
                                    np.array([true_lon, true_lat]), pointing_direction=pointing_direction)*u.deg).to(u.cm**2).value)
        
    def log_edisp(self, recon_energy, true_energy, true_lon, true_lat, pointing_direction=None):
        """Wrapper for the Gammapy interpretation of the CTA point spread function.

        Args:
            recon_energy (float): Measured energy value by the CTA
            true_energy (float): True energy of a gamma-ray event detected by the CTA
            true_lon (float): True FOV longitude of a gamma-ray event 
                detected by the CTA
            true_lat (_type_): True FOV latitude of a gamma-ray event 
                detected by the CTA

        Returns:
            float: natural log of the CTA energy dispersion likelihood for the given 
                gamma-ray event data
        """
        return np.log(self.edisp_default.evaluate(energy_true=true_energy*u.TeV,
                                                        migra = recon_energy/true_energy, 
                                                        offset=convertlonlat_to_offset(
                                                            np.array([true_lon, true_lat]), 
                                                            pointing_direction=pointing_direction)*u.deg).value)


    def log_psf(self, recon_lon, recon_lat, true_energy, true_lon, true_lat, pointing_direction=None):
        """Wrapper for the Gammapy interpretation of the CTA point spread function.

        Args:
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
            float: natural log of the CTA point spread function likelihood for the given 
                gamma-ray event data
        """
        reconstructed_spatialcoord = np.array([recon_lon, recon_lat])
        truespatialcoord = np.array([true_lon, true_lat])
        rad = angularseparation(reconstructed_spatialcoord, truespatialcoord).flatten()
        offset  = convertlonlat_to_offset(truespatialcoord, pointing_direction=pointing_direction).flatten()
        output = np.log(self.psf_default.evaluate(energy_true=true_energy*u.TeV,
                                                        rad = rad*u.deg, 
                                                        offset=offset*u.deg).value)
        
        return output


    def single_loglikelihood(self, recon_energy, recon_lon, recon_lat, true_energy, true_lon, true_lat, pointing_direction=None):
        """Wrapper for the Gammapy interpretation of the CTA IRFs to output the log 
            likelihood values for the given gamma-ray event data

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
        reconstructed_spatialcoord = np.array([recon_lon, recon_lat])
        truespatialcoord = np.array([true_lon, true_lat])
        rad = angularseparation(reconstructed_spatialcoord, truespatialcoord).flatten()
        offset  = convertlonlat_to_offset(truespatialcoord, pointing_direction=pointing_direction).flatten()
        output  = np.log(self.psf_default.evaluate(energy_true=true_energy*u.TeV,
                                                        rad = rad*u.deg, 
                                                        offset=offset*u.deg).value)
        
        output  += np.log(self.edisp_default.evaluate(energy_true=true_energy*u.TeV,
                                                        migra = recon_energy/true_energy, 
                                                        offset=offset*u.deg).value)
        return output


    def dynesty_single_loglikelihood(self, true_vals, recon_energy, recon_lon, recon_lat, pointing_direction=None):
        """Wrapper for the Gammapy interpretation of the CTA IRFs to output the log 
            likelihood values for the given gamma-ray event data

        Args:
            true_energy (float): True energy of a gamma-ray event detected by the CTA
            true_lon (float): True FOV longitude of a gamma-ray event 
                detected by the CTA
            true_lat (float): True FOV latitude of a gamma-ray event 
                detected by the CTA
            recon_energy (float): Measured energy value by the CTA
            recon_lon (float): Measured FOV longitude of a gamma-ray event
                detected by the CTA
            recon_lat (float): Measured FOV latitude of a gamma-ray event
                detected by the CTA

        Returns:
            float: natural log of the full CTA likelihood for the given gamma-ray 
                event data
        """
        return self.single_loglikelihood(recon_energy, recon_lon, recon_lat, *true_vals, pointing_direction=None)