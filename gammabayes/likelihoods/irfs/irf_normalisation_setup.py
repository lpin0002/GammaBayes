# If running from main file, the terminal format should be $ python -m gammabayes.utils.default_file_setup 1 1 1
    # If you're running from a script there shouldn't be any issues as setup is just a func
from gammabayes.utils.event_axes import log10eaxistrue, longitudeaxistrue, latitudeaxistrue, log10eaxis, longitudeaxis, latitudeaxis, logjacob
from gammabayes.utils.utils import angularseparation, convertlonlat_to_offset, resources_dir
from tqdm import tqdm

from gammabayes.likelihoods.irfs.gammapy_wrappers import irfs, log_edisp, log_psf

import numpy as np
from astropy import units as u
from scipy import special
from scipy.integrate import simps
import os, sys


aeff = irfs['aeff']

aefffunc = lambda energy, offset: aeff.evaluate(energy_true = energy*u.TeV, offset=offset*u.deg).to(u.cm**2).value



def irf_norm_setup(log10eaxistrue=log10eaxistrue, log10eaxis=log10eaxis, 
          longitudeaxistrue=longitudeaxistrue, longitudeaxis=longitudeaxis, latitudeaxistrue=latitudeaxistrue, latitudeaxis=latitudeaxis,
          logjacob=logjacob, save_directory = resources_dir, psf=log_psf, edisp=log_edisp,
          save_results=True, outputresults=False):
    """Produces default IRF normalisation matrices

    Args:
        log10eaxistrue (np.ndarray, optional): Dicrete true log10 energy values
            of CTA event data. Defaults to log10eaxistrue.

        log10eaxis (np.ndarray, optional): Dicrete measured log10 energy values
            of CTA event data. Defaults to log10eaxis.

        longitudeaxistrue (np.ndarray, optional): Dicrete true fov longitude values
            of CTA event data. Defaults to longitudeaxistrue.

        longitudeaxis (np.ndarray, optional): Dicrete measured fov longitude values
            of CTA event data. Defaults to longitudeaxis.

        latitudeaxistrue (np.ndarray, optional): Dicrete true fov latitude values
            of CTA event data. Defaults to latitudeaxistrue.

        latitudeaxis (np.ndarray, optional): Dicrete measured fov latitude values
            of CTA event data. Defaults to latitudeaxis.

        logjacob (np.ndarray, optional): _description_. Defaults to logjacob.

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
    for logeval in tqdm(log10eaxistrue, desc='Setting up psf normalisation', ncols=80):
        psflogerow = []
        for lonval in longitudeaxistrue:
            log10eaxistrue_mesh, longitudeaxistrue_mesh, latitudeaxistrue_mesh, longitudeaxis_mesh, latitudeaxis_mesh  = np.meshgrid(logeval,
                                                                                                                                    lonval, 
                                                                                                                                    latitudeaxistrue, 
                                                                                                                                    longitudeaxis, 
                                                                                                                                    latitudeaxis, indexing='ij')

            # truecoords = np.array([longitudeaxistrue_mesh.flatten(), latitudeaxistrue_mesh.flatten()])

            # recon_coords = np.array([longitudeaxis_mesh.flatten(), latitudeaxis_mesh.flatten()])

            # rad = angularseparation(recon_coords, truecoords)
            # offset = convertlonlat_to_offset(truecoords)
            psfvals = log_psf(longitudeaxis_mesh.flatten(), latitudeaxis_mesh.flatten(), 
                                log10eaxistrue_mesh.flatten(), longitudeaxistrue_mesh.flatten(), latitudeaxistrue_mesh.flatten()).reshape(log10eaxistrue_mesh.shape)

            # psfvals = psf(rad, log10eaxistrue_mesh.flatten(), offset).reshape(log10eaxistrue_mesh.shape)
            psfnormvals = special.logsumexp(psfvals, axis=(-2,-1))
            
            psflogerow.append(psfnormvals)
        psfnorm.append(psflogerow)
            

# 
    psfnorm = np.squeeze(np.array(psfnorm))



    edispnorm = []
    for logeval in tqdm(log10eaxistrue, desc='Setting up edisp normalisation', ncols=80):
        log10eaxistrue_mesh, longitudeaxistrue_mesh, latitudeaxistrue_mesh, log10eaxis_mesh  = np.meshgrid(logeval,
                                                                                                            longitudeaxistrue, 
                                                                                                            latitudeaxistrue, 
                                                                                                            log10eaxis,
                                                                                                            indexing='ij')

        # truecoords = np.array([longitudeaxistrue_mesh.flatten(), latitudeaxistrue_mesh.flatten()])
        
        # offset = convertlonlat_to_offset(truecoords)

        # edispvals = np.squeeze(edisp(log10eaxis_mesh.flatten(), log10eaxistrue_mesh.flatten(), offset).reshape(log10eaxistrue_mesh.shape))
        edispvals = np.squeeze(log_edisp(log10eaxis_mesh.flatten(), 
                                        log10eaxistrue_mesh.flatten(), longitudeaxistrue_mesh.flatten(), latitudeaxistrue_mesh.flatten()).reshape(log10eaxistrue_mesh.shape))
        edispnormvals = special.logsumexp(edispvals+logjacob, axis=-1)
        
        edispnorm.append(edispnormvals)


    edispnorm = np.array(edispnorm)

    edispnorm[np.isneginf(edispnorm)] = 0
    psfnorm[np.isneginf(psfnorm)] = 0

    if save_results:
        np.save(save_directory+"/psfnormalisation.npy", psfnorm)
        np.save(save_directory+"/edispnormalisation.npy", edispnorm)

        
    if outputresults:
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