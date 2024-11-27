from gammabayes import haversine, resources_dir
import numpy as np
from astropy import units as u
from astropy.units import Quantity
from gammapy.irf import load_irf_dict_from_file
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
import os
import zipfile

from ..CTAO_IRFs.CTAO_irf_file_utils import find_ctao_irf_file_path
from ..HESS_IRFs import extract_and_generate_hess_data
from gammapy.data import DataStore
import importlib.resources as pkg_resources
import time

class IRFExtractor(object):
    def __init__(self, 
                 zenith_angle:int=20, 
                 hemisphere:str='South', 
                 prod_vers=5, 
                 observation_time: float|u.Quantity = 50*u.hr,
                 file_path: str=None,
                 psf_units: u.Unit = (1/u.deg**2).unit,
                 edisp_units: u.Unit = (1/u.TeV).unit,
                 aeff_units: u.Unit = u.cm**2,
                 CCR_BKG_units: u.Unit = (1/(u.deg**2*u.TeV*u.s)).unit,
                 instrument: str ='CTAO',
                 obs_id: int = None,
                 pointing_dir: np.ndarray[u.Quantity] = np.array([0, 0])*u.deg):

        self._format_instrument(instrument)


        self.file_path = file_path
        self.observation_time = observation_time
        if hasattr(self.observation_time, "unit"):
            irf_time_in_seconds = self.observation_time.to(u.s)


        else:
            irf_time_in_seconds = 180000*u.s


        if self.file_path is None:
            # self.file_path = resources_dir+f'/irf_fits_files/Prod5-South-20deg-AverageAz-14MSTs37SSTs.{irf_time_in_seconds}s-v0.1.fits'

            if self.instrument in ['CTA', 'CTAO']:
                self._CTAO_init(zenith_angle=zenith_angle, hemisphere=hemisphere, prod_vers=prod_vers)
            else:
                self._HESS_init(pointing_dir=pointing_dir, obs_id=obs_id)


        else:
            self.extracted_default_irfs  = load_irf_dict_from_file(self.file_path)


        self.psf_units = psf_units
        self.edisp_units = edisp_units
        self.aeff_units = aeff_units
        self.CCR_BKG_units = CCR_BKG_units


        self.edisp_default      = self.extracted_default_irfs['edisp']
        # self.edisp_default.normalize()

        self.psf_default        = self.extracted_default_irfs['psf']


        if hasattr(self.psf_default, 'to_psf3d'):
            self.psf3d              = self.psf_default.to_psf3d()
        else:
            self.psf3d              = self.psf_default

        # self.psf3d.normalize()

        self.aeff_default       = self.extracted_default_irfs['aeff']
        self.CCR_BKG       = self.extracted_default_irfs['bkg'].to_2d()

        self.zenith = zenith_angle
        self.hemisphere = hemisphere
        self.prod_vers = prod_vers


    def _format_instrument(self, instrument_str:str):
        
        if instrument_str in ['CTA', 'CTAO']:
            self.instrument = 'CTAO'
        elif instrument_str == 'HESS':
            self.instrument = instrument_str
        else:
            raise ValueError("You have specified an instrument that GammaBayes currently does not support. Please select either CTAO or HESS")





    def _CTAO_init(self, zenith_angle, hemisphere, prod_vers, *args, **kwargs):
        prod_version, fits_file_path = find_ctao_irf_file_path(zenith_angle=zenith_angle, 
                                            hemisphere=hemisphere, 
                                            prod_vers=prod_vers)
        
        # print(f"\nPath to irf fits file: {fits_file_path}\n")
        self.extracted_default_irfs  = load_irf_dict_from_file(fits_file_path)



    def _HESS_init(self, obs_id, pointing_dir):
        extract_and_generate_hess_data()
        package_data_dir = pkg_resources.files('gammabayes').joinpath('package_data')

        data_store = DataStore.from_dir(package_data_dir / "HESS_DL3_DR1")

        if not obs_id is None:
            obs = data_store.obs(obs_id)

        else:
            # Find the observation with the closest pointing direction
            obs_ids = []
            distances = []
            for obs_id, lon,lat in data_store.obs_table[['OBS_ID', 'GLON_PNT', 'GLAT_PNT', ]]:

                if lon>180:
                    lon = lon - 360

                obs_ids.append(obs_id)
                distances.append(np.sqrt((lon-pointing_dir.value[0])**2+(lat-pointing_dir.value[1])**2))


            obs = data_store.obs(obs_ids[np.array(distances).argmin()])


        
        self.extracted_default_irfs = {method: getattr(obs, method) for method in obs.available_irfs}






    # Deprecated function that I need to get rid of eventually
    def aefffunc(self, energy:Quantity, offset:Quantity, parameters:dict={}):
        """Deprecated function that I need to safely get rid of eventually.

        Args:
            energy (Quantity): _description_
            offset (Quantity): _description_
            parameters (dict, optional): _description_. Defaults to {}.

        Returns:
            _type_: _description_
        """
        return self.aeff_default.evaluate(energy_true = energy, offset=offset).to(self.aeff_units)

    def log_aeff(self, energy, lon, lat, pointing_dir=[0*u.deg,0*u.deg], parameters={}):
        """
        Wrapper for the Gammapy interpretation of the log of the CTA effective area function.

        Args:
            energy (Quantity): True energy of a gamma-ray event detected by the CTA.
            lon (Quantity): True FOV longitude of a gamma-ray event detected by the CTA.
            lat (Quantity): True FOV latitude of a gamma-ray event detected by the CTA.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: The natural log of the effective area of the CTA in cm^2.
        """
        return np.log(self.aeff_default.evaluate(energy_true = energy, 
                                offset=haversine(
                                    lon, lat, pointing_dir[0], pointing_dir[1])).to(self.aeff_units).value)
        
    def log_edisp(self, recon_energy:Quantity, 
                  true_energy:Quantity, true_lon:Quantity, true_lat:Quantity, 
                  pointing_dir:list[Quantity]=[0*u.deg,0*u.deg], parameters:dict={}, migration_cut=100):
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


        offset = haversine(true_lon, true_lat, pointing_dir[0], pointing_dir[1])

        migration = recon_energy/true_energy



        edisp_val = np.where(np.logical_and(migration<migration_cut, migration>1/migration_cut), self.edisp_default.evaluate(energy_true=true_energy,
                                                        migra = migration, 
                                                        offset=offset), 0)

        adjusted_edisp_val = edisp_val/(true_energy.unit)

        adjusted_edisp_val = (adjusted_edisp_val).to(self.edisp_units).value


        # edisp output is dimensionless when it should have units of 1/TeV
        log_output = np.log(adjusted_edisp_val)

        return log_output



    def log_psf(self, recon_lon:Quantity, recon_lat:Quantity, 
                true_energy:Quantity, true_lon:Quantity, true_lat:Quantity, 
                pointing_dir:list[Quantity]=[0*u.deg,0*u.deg], parameters:dict={}):
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

        rad = haversine(recon_lon.flatten(), recon_lat.flatten(), true_lon.flatten(), true_lat.flatten(),).flatten()

        offset  = haversine(true_lon.flatten(), true_lat.flatten(), pointing_dir[0], pointing_dir[1]).flatten()

        output = np.log(self.psf_default.evaluate(energy_true=true_energy, rad = rad, 
                                                  offset=offset).to(self.psf_units).value)
                
        return output


    # Made for the reverse convention of parameter order required by dynesty
    def dynesty_single_loglikelihood(self, 
                                     true_vals: list[Quantity]|tuple[Quantity], 
                                     recon_energy:Quantity, recon_lon:Quantity, recon_lat:Quantity, 
                                     pointing_dir:list[Quantity]=[0*u.deg,0*u.deg], parameters:dict={}):
        """
        Wrapper for the Gammapy interpretation of the CTA IRFs to output the log likelihood values 
        for the given gamma-ray event data for use with dynesty as a likelihood.

        Args:
            true_vals (list[Quantity] | tuple[Quantity]): True values of the gamma-ray event (true_energy, true_lon, true_lat).
            recon_energy (Quantity): Measured energy value by the CTA.
            recon_lon (Quantity): Measured FOV longitude of a gamma-ray event detected by the CTA.
            recon_lat (Quantity): Measured FOV latitude of a gamma-ray event detected by the CTA.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: Natural log of the full CTA likelihood for the given gamma-ray event data.
        """
        true_energy, true_lon, true_lat = true_vals
        offset  = haversine(true_lon, true_lat, pointing_dir[0], pointing_dir[1])

        output = np.log(self.edisp_default.evaluate(energy_true=true_energy,
                                                        migra = recon_energy/true_energy, 
                                                        offset=offset))

        rad = haversine(recon_lon, recon_lat, true_lon, true_lat)

        output+=  np.log(self.psf_default.evaluate(energy_true=true_energy,
                                                        rad = rad, 
                                                        offset=offset).to(self.psf_units))
        
        return output
    
    def log_bkg_CCR(self, energy:Quantity, lon:Quantity, lat:Quantity, 
                    spectral_parameters:dict={}, spatial_parameters:dict={},
                    pointing_dir:list[Quantity]=[0*u.deg,0*u.deg], ):
        """
        Wrapper for the Gammapy interpretation of the log of the CTA's background charged cosmic-ray mis-identification rate.

        Args:
            energy (Quantity): True energy of a gamma-ray event detected by the CTA.
            lon (Quantity): True FOV longitude of a gamma-ray event detected by the CTA.
            lat (Quantity): True FOV latitude of a gamma-ray event detected by the CTA.
            spectral_parameters (dict, optional): Spectral parameters. Defaults to {}.
            spatial_parameters (dict, optional): Spatial parameters. Defaults to {}.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: Natural log of the charged cosmic ray mis-identification rate for the CTA.
        """

        offset  = haversine(lon, lat, pointing_dir[0], pointing_dir[1])



        return np.log(self.CCR_BKG.evaluate(energy=energy, offset=offset).to(self.CCR_BKG_units).value)
    

    
