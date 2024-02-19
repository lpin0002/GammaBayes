from gammabayes.utils import resources_dir, haversine
import numpy as np
from astropy import units as u
from gammapy.irf import load_irf_dict_from_file,load_cta_irfs
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
from gammabayes.utils import resources_dir
import os
import zipfile


def find_file_in_parent(parent_folder, end_of_filename):
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(end_of_filename):
                return os.path.join(root, file)
    return None



def find_irf_file_path(zenith_angle=20, hemisphere='South', prod_vers=5, subarray=None):
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


    if prod_vers == 'prod5-v0.1':
        untarred_parent_folder= parent_irf_folder+f"CTA-Performance-{prod_vers}-{hemisphere}-{subarray}{zenith_angle}deg.FITS"
        end_of_filename = '180000s-v0.1.fits.gz'

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
    def __init__(self, zenith_angle, hemisphere, prod_vers=5, file_path=None):
        self.file_path = file_path
        if self.file_path is None:
            self.file_path = resources_dir+'/irf_fits_files/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits'

            if (zenith_angle is None) and (hemisphere is None):
                extracted_default_irfs  = load_irf_dict_from_file(self.file_path)
            else:
                prod_version, fits_file_path = find_irf_file_path(zenith_angle=zenith_angle, 
                                                    hemisphere=hemisphere, 
                                                    prod_vers=prod_vers)
                
                print(f"\nPath to irf fits file: {fits_file_path}\n")
                if prod_version=='prod5':
                    extracted_default_irfs  = load_irf_dict_from_file(fits_file_path)
                elif prod_version=='prod3b':
                    extracted_default_irfs  = load_cta_irfs(fits_file_path)


        else:
            extracted_default_irfs  = load_irf_dict_from_file(self.file_path)


        self.edisp_default      = extracted_default_irfs['edisp']
        self.psf_default        = extracted_default_irfs['psf']

        self.psf3d              = self.psf_default.to_psf3d()
        self.aeff_default       = extracted_default_irfs['aeff']



    # Deprecated function that I need to get rid of eventually
    def aefffunc(self, energy, offset, parameters={}):
        return self.aeff_default.evaluate(energy_true = energy*u.TeV, offset=offset*u.deg).to(u.cm**2).value

    def log_aeff(self, energy, longitude, latitude, pointing_direction=[0,0], parameters={}):
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
        return np.log(self.aeff_default.evaluate(energy_true = energy*u.TeV, 
                                offset=haversine(
                                    longitude, latitude, pointing_direction[0], pointing_direction[1])*u.deg).to(u.cm**2).value)
        
    def log_edisp(self, recon_energy, true_energy, true_lon, true_lat, pointing_direction=[0,0], parameters={}):
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
                                                        offset=haversine(
                                    true_lon, true_lat, pointing_direction[0], pointing_direction[1])*u.deg).value)


    def log_psf(self, recon_lon, recon_lat, true_energy, true_lon, true_lat, pointing_direction=[0,0], parameters={}):
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
        rad = haversine(recon_lon.flatten(), recon_lat.flatten(), true_lon.flatten(), true_lat.flatten(),).flatten()
        offset  = haversine(true_lon.flatten(), true_lat.flatten(), pointing_direction[0], pointing_direction[1]).flatten()
        output = np.log(self.psf_default.evaluate(energy_true=true_energy*u.TeV,
                                                        rad = rad*u.deg, 
                                                        offset=offset*u.deg).value)
        
        return output


    def single_loglikelihood(self, 
                             recon_energy, recon_lon, recon_lat, 
                             true_energy, true_lon, true_lat, 
                             pointing_direction=[0,0], parameters={}):
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
        offset  = haversine(true_lon.flatten(), true_lat.flatten(), pointing_direction[0], pointing_direction[1]).flatten()

        
        # # Unique edisp input filtering
        # flatten_edisp_param_vals = np.array([recon_energy.flatten(), true_energy.flatten(), offset])
        # unique_param_vals = np.unique(flatten_edisp_param_vals, axis=1)

        output = np.log(self.edisp_default.evaluate(energy_true=true_energy.flatten()*u.TeV,
                                                        migra = recon_energy.flatten()/true_energy.flatten(), 
                                                        offset=offset*u.deg).value)

        # mask = np.all(unique_param_vals[:, None, :] == flatten_edisp_param_vals[:, :, None], axis=0)
        # slices = np.where(mask, unique_edisp_output[None, :], 0.0)

        # output = np.sum(slices, axis=-1).reshape(recon_energy.shape)
                
        
        # Standard input method for psf (for now)
        rad = haversine(recon_lon.flatten(), recon_lat.flatten(), true_lon.flatten(), true_lat.flatten(),).flatten()

        output+=  np.log(self.psf_default.evaluate(energy_true=true_energy*u.TeV,
                                                        rad = rad*u.deg, 
                                                        offset=offset*u.deg).value)
        return output.reshape(true_lon.shape)

    # Made for the reverse convention of parameter order required by dynesty
    def dynesty_single_loglikelihood(self, 
                                     true_vals, 
                                     recon_energy, recon_lon, recon_lat, 
                                     pointing_direction=[0,0], parameters={}):
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
        true_energy, true_lon, true_lat = true_vals
        offset  = haversine(true_lon, true_lat, pointing_direction[0], pointing_direction[1])

        output = np.log(self.edisp_default.evaluate(energy_true=true_energy*u.TeV,
                                                        migra = recon_energy/true_energy, 
                                                        offset=offset*u.deg).value)

        rad = haversine(recon_lon, recon_lat, true_lon, true_lat)

        output+=  np.log(self.psf_default.evaluate(energy_true=true_energy*u.TeV,
                                                        rad = rad*u.deg, 
                                                        offset=offset*u.deg).value)
        
        return output