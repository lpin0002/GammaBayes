from gammabayes import haversine, resources_dir
import numpy as np
from astropy import units as u
from astropy.units import Quantity
from gammapy.irf import load_irf_dict_from_file,load_cta_irfs
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
import os
import zipfile


def find_file_in_parent(parent_folder:str, end_of_filename:str):
    """
    Searches for a file with a specific ending in a parent folder.

    Args:
        parent_folder (str): The parent directory to search in.
        end_of_filename (str): The file name ending to search for.

    Returns:
        str | None: The full path to the file if found, otherwise None.
    """

    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(end_of_filename):
                return os.path.join(root, file)
    return None



def find_irf_file_path(zenith_angle:int=20, hemisphere:str='South', prod_vers=5, irf_time_seconds = 18000, subarray:str=None):
    """
    Finds the file path for the CTA IRF file based on given parameters.

    Args:
        zenith_angle (int, optional): Zenith angle. Defaults to 20.
        hemisphere (str, optional): Hemisphere ('North' or 'South'). Defaults to 'South'.
        prod_vers (int, optional): Production version (3 or 5). Defaults to 5.
        subarray (str, optional): Subarray type (e.g., 'SST', 'MST', 'LST'). Defaults to None.

    Raises:
        Exception: If invalid input values are provided.

    Returns:
        tuple[str, str]: The IRF version and the file path.
    """

    prod5_variations = [5,'prod5', '5', 'prod5-v0.1']
    prod3_variations = [3,'prod3', '3', '3b', 'prod3b']


    if prod_vers in prod5_variations:
        parent_irf_folder = resources_dir+"/irf_fits_files/prod5/"
        prod_vers = 'prod5-v0.1'
    elif prod_vers in prod3_variations:
        parent_irf_folder = resources_dir+"/irf_fits_files/prod3b/"
        prod_vers = 'prod3b'
    else:
        raise Exception(f"Invalid irf version given must, be one of {prod5_variations} or {prod3_variations}.")


    hemisphere_south_variations = ['s','south']
    hemisphere_north_variations = ['n','north']

    if hemisphere.lower() in hemisphere_south_variations:
        hemisphere = 'South'
    elif hemisphere.lower() in hemisphere_north_variations:
        hemisphere = 'North'
    else:
        raise Exception(f"Invalid hemisphere input given, must be one of {hemisphere_south_variations} or {hemisphere_north_variations}.")
    
    possible_zenith_angles = [20, 40, 60]

    if int(zenith_angle) in possible_zenith_angles:
        zenith_angle = str(int(zenith_angle))
    else:
        raise Exception(f"Invalid zenith angle given, must be one of {possible_zenith_angles}.")
    

    possible_subarrays = ['sstsubarray', 'mstsubarray', 'lstsubarray', 'sst', 'mst', 'lst']
    formatted_subarray_types = ['SSTSubArray', 'MSTSubArray', 'LSTSubArray', 'SSTSubArray', 'MSTSubArray', 'LSTSubArray']

    if not(subarray is None):
        if subarray.lower() in possible_subarrays:
            subarray = formatted_subarray_types[possible_subarrays.index(subarray.lower())]
            subarray = subarray+'-'
        else:
            raise Exception(f'Invalid sub array type given. Must be one of {formatted_subarray_types}.')
    else:
        subarray = ''

    irf_times = np.array([1800, 18000, 180000])

    time = irf_times[np.abs(irf_time_seconds-irf_times).argmin()]



    if prod_vers == 'prod5-v0.1':
        untarred_parent_folder= parent_irf_folder+f"CTA-Performance-{prod_vers}-{hemisphere}-{subarray}{zenith_angle}deg.FITS"
        end_of_filename = f'{time}s-v0.1.fits.gz'

        filename = find_file_in_parent(untarred_parent_folder, end_of_filename)

        if filename:
            return 'prod5', filename
        else:
            print("Stem file has not been un-tarred.")
            #TODO: Unzip irf files in some sort of setup function on install
            import tarfile 
            
            # open file 
            try:
                file = tarfile.open(untarred_parent_folder+'.tar.gz') 
                # extracting file 
                file.extractall(untarred_parent_folder)

                file.close() 

                filename = find_file_in_parent(untarred_parent_folder, end_of_filename)

                if filename:
                    return 'prod5', filename
                else:
                    raise Exception('Found tarred file but could not extract sub-file.')
            except:
                raise Exception("Could not find prod5 irf file for the given inputs.")
    elif prod_vers == 'prod3b':

        unzipped_parent_folder = parent_irf_folder+f"{hemisphere}_z{zenith_angle}_50h"

        filename = find_file_in_parent(unzipped_parent_folder, '.fits')

        if filename:
            return 'prod3b', filename
        else:
            zip_filename = unzipped_parent_folder+'.zip'

            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                dirname = os.path.dirname(unzipped_parent_folder)
                zip_ref.extractall(dirname)

            filename = find_file_in_parent(unzipped_parent_folder, '.fits')

            print(filename)

            if filename:
                return 'prod3b', filename
            else:
                raise Exception('Found ziiped file but could not extract sub-file.')



class IRFExtractor(object):
    def __init__(self, 
                 zenith_angle:int, 
                 hemisphere:str, 
                 prod_vers=5, 
                 observation_time: float|u.Quantity = 50*u.hr,
                 file_path: str=None,
                 psf_units: u.Unit = (1/u.deg**2).unit,
                 edisp_units: u.Unit = (1/u.TeV).unit,
                 aeff_units: u.Unit = u.cm**2,
                 CCR_BKG_units: u.Unit = (1/(u.deg**2*u.TeV*u.s)).unit):




        self.file_path = file_path
        self.observation_time = observation_time
        if hasattr(self.observation_time, "unit"):
            irf_time_in_seconds = self.observation_time.to(u.s)


        else:
            irf_time_in_seconds = 180000


        if self.file_path is None:
            # self.file_path = resources_dir+f'/irf_fits_files/Prod5-South-20deg-AverageAz-14MSTs37SSTs.{irf_time_in_seconds}s-v0.1.fits'

            if (zenith_angle is None) and (hemisphere is None):
                self.extracted_default_irfs  = load_irf_dict_from_file(self.file_path)
            else:
                prod_version, fits_file_path = find_irf_file_path(zenith_angle=zenith_angle, 
                                                    hemisphere=hemisphere, 
                                                    prod_vers=prod_vers)
                
                # print(f"\nPath to irf fits file: {fits_file_path}\n")
                if prod_version=='prod5':
                    self.extracted_default_irfs  = load_irf_dict_from_file(fits_file_path)
                elif prod_version=='prod3b':
                    self.extracted_default_irfs  = load_cta_irfs(fits_file_path)


        else:
            self.extracted_default_irfs  = load_irf_dict_from_file(self.file_path)


        self.psf_units = psf_units
        self.edisp_units = edisp_units
        self.aeff_units = aeff_units
        self.CCR_BKG_units = CCR_BKG_units


        self.edisp_default      = self.extracted_default_irfs['edisp']
        # self.edisp_default.normalize()

        self.psf_default        = self.extracted_default_irfs['psf']

        self.psf3d              = self.psf_default.to_psf3d()
        # self.psf3d.normalize()

        self.aeff_default       = self.extracted_default_irfs['aeff']
        self.CCR_BKG       = self.extracted_default_irfs['bkg'].to_2d()



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
                  pointing_dir:list[Quantity]=[0*u.deg,0*u.deg], parameters:dict={}):
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

        # edisp output is dimensionless when it should have units of 1/TeV
        return np.log((self.edisp_default.evaluate(energy_true=true_energy,
                                                        migra = recon_energy/true_energy, 
                                                        offset=offset)/true_energy).to(self.edisp_units).value)


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

        offset = offset

        output = np.log(self.psf_default.evaluate(energy_true=true_energy,
                                                        rad = rad, 
                                                        offset=offset).to(self.psf_units).value)
                
        return output


    def single_loglikelihood(self, 
                             recon_energy:Quantity, recon_lon:Quantity, recon_lat:Quantity, 
                             true_energy:Quantity, true_lon:Quantity, true_lat:Quantity, 
                             pointing_dir:list[Quantity]=[0*u.deg,0*u.deg], parameters:dict={}):
        """
        Wrapper for the Gammapy interpretation of the CTA IRFs to output the log likelihood values 
        for the given gamma-ray event datum.

        Args:
            recon_energy (Quantity): Measured energy value by the CTA.
            recon_lon (Quantity): Measured FOV longitude of a gamma-ray event detected by the CTA.
            recon_lat (Quantity): Measured FOV latitude of a gamma-ray event detected by the CTA.
            true_energy (Quantity): True energy of a gamma-ray event detected by the CTA.
            true_lon (Quantity): True FOV longitude of a gamma-ray event detected by the CTA.
            true_lat (Quantity): True FOV latitude of a gamma-ray event detected by the CTA.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: Natural log of the full CTA likelihood for the given gamma-ray event data.
        """

        output = self.log_edisp(recon_energy, 
                                true_energy, true_lon, true_lat, 
                                pointing_dir)


        output +=  self.log_psf(recon_lon, recon_lat, 
                                true_energy, true_lon, true_lat, 
                                pointing_dir)
        
        return output.reshape(true_lon.shape)

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
    

    
