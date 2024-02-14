# If running from main file, the terminal format should be $ python -m gammabayes.utils.default_file_setup 1 1 1
    # If you're running from a script there shouldn't be any issues as setup is just a func
from gammabayes.utils import resources_dir, iterate_logspace_integration
from gammabayes.likelihoods.irfs.prod5.gammapy_wrappers import log_edisp, log_psf


from tqdm import tqdm
import numpy as np
from astropy import units as u
from scipy import special
from scipy.integrate import simps
import os, sys


def irf_norm_setup(energy_true_axis, energy_recon_axis, 
          longitudeaxistrue, longitudeaxis, 
          latitudeaxistrue, latitudeaxis,
          save_directory = resources_dir, log_psf=log_psf, log_edisp=log_edisp,
          save_results=False):
    """Produces default IRF normalisation matrices

    Args:
        energy_true_axis (np.ndarray, optional): Dicrete true energy values
            of CTA event data. Defaults to log10eaxistrue.

        energy_recon_axis (np.ndarray, optional): Dicrete measured energy values
            of CTA event data. Defaults to log10eaxis.

        longitudeaxistrue (np.ndarray, optional): Dicrete true fov longitude values
            of CTA event data. Defaults to longitudeaxistrue.

        longitudeaxis (np.ndarray, optional): Dicrete measured fov longitude values
            of CTA event data. Defaults to longitudeaxis.

        latitudeaxistrue (np.ndarray, optional): Dicrete true fov latitude values
            of CTA event data. Defaults to latitudeaxistrue.

        latitudeaxis (np.ndarray, optional): Dicrete measured fov latitude values
            of CTA event data. Defaults to latitudeaxis.

        save_directory (str, optional): Path to save results. Defaults to resources_dir.

        logpsf (func, optional): Function representing the log point spread 
        function for the CTA. Defaults to psf_test.

        logedisp (func, optional): Function representing the log energy dispersion
          for the CTA. Defaults to edisp_test.

        save_results (bool, optional): _description_. Defaults to True.

        outputresults (bool, optional): _description_. Defaults to False.
    """

    
    print(f"Save directory is {save_directory}")


    psfnorm = []
    for energy_val in tqdm(energy_true_axis, desc='Setting up psf normalisation', ncols=80):
        psflogerow = []
        for lonval in longitudeaxistrue:
            energy_true_axis_mesh, longitude_axis_true_mesh, latitude_axis_true_mesh, longitude_axis_mesh, latitude_axis_mesh  = np.meshgrid(energy_val,
                                                                                                                                    lonval, 
                                                                                                                                    latitudeaxistrue, 
                                                                                                                                    longitudeaxis, 
                                                                                                                                    latitudeaxis, indexing='ij')

            psfvals = log_psf(longitude_axis_mesh.flatten(), latitude_axis_mesh.flatten(), 
                                energy_true_axis_mesh.flatten(), 
                                longitude_axis_true_mesh.flatten(), 
                                latitude_axis_true_mesh.flatten(),).reshape(energy_true_axis_mesh.shape)
            psfnormvals = iterate_logspace_integration(np.squeeze(psfvals), axes=[longitudeaxis, latitudeaxis], axisindices=[1,2])
            
            psflogerow.append(psfnormvals)
        psfnorm.append(psflogerow)

    edispnorm = []
    for energy_val in tqdm(energy_true_axis, desc='Setting up edisp normalisation', ncols=80):
        energy_true_axis_mesh, longitude_axis_true_mesh, latitude_axis_true_mesh, energy_recon_axis_mesh  = np.meshgrid(energy_val,
                                                                                                            longitudeaxistrue, 
                                                                                                            latitudeaxistrue, 
                                                                                                            energy_recon_axis,
                                                                                                            indexing='ij')
        edispvals = np.squeeze(log_edisp(energy_recon_axis_mesh.flatten(), 
                                        energy_true_axis_mesh.flatten(), 
                                        longitude_axis_true_mesh.flatten(), 
                                        latitude_axis_true_mesh.flatten(),).reshape(energy_true_axis_mesh.shape))
        edispnormvals = iterate_logspace_integration(np.squeeze(edispvals), axes=[energy_recon_axis], axisindices=[2])
        
        edispnorm.append(edispnormvals)


    psfnorm = np.squeeze(np.array(psfnorm))
    edispnorm = np.array(edispnorm)

    edispnorm[np.isneginf(edispnorm)] = 0
    psfnorm[np.isneginf(psfnorm)] = 0
    if save_results:
        np.save(save_directory+"/log_prod5_psf_normalisations.npy", psfnorm)
        np.save(save_directory+"/log_prod5_edisp_normalisations.npy", edispnorm)

    return psfnorm, edispnorm


if __name__=="__main__":
    try:
        save_directory = sys.argv[1]
    except:
        save_directory = 0

    if save_directory!=0:
        print(f"Chosen save directory is: {save_directory}")
        irf_norm_setup(save_directory=save_directory)
    else:
        print("No save directory specified.")

        irf_norm_setup()