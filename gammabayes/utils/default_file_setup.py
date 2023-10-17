from os import path
import sys, os
sys.path.append(path.dirname(path.dirname(path.dirname(os.path.abspath(__file__)))))


from gammabayes.utils.load_package_defaults import log10eaxistrue, longitudeaxistrue, latitudeaxistrue, log10eaxis, longitudeaxis, latitudeaxis, logjacob
from gammabayes.utils.utils import angularseparation, convertlonlat_to_offset
from gammabayes.utils.utils import tqdm, resources_dir, edisp_test, psf_test #, psf_efficient, edisp_efficient, 
from gammabayes.utils.utils import irfs

from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, WcsGeom
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy import special
from scipy.integrate import simps
import os, sys
from multiprocessing import Pool, freeze_support

from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
    create_fermi_isotropic_diffuse_model,
)

from gammapy.catalog import SourceCatalogHGPS


aeff = irfs['aeff']

aefffunc = lambda energy, offset: aeff.evaluate(energy_true = energy*u.TeV, offset=offset*u.deg).to(u.cm**2).value


try:
    run_function = int(sys.argv[1])
except:
    run_function = 0
    
try:
    setup_irfnormalisations = int(sys.argv[2])
except:
    setup_irfnormalisations = 1
    
try:
    setup_astrobkg = int(sys.argv[3])
except:
    setup_astrobkg = 1

def default_file_setup(setup_irfnormalisations=1, setup_astrobkg=1, log10eaxistrue=log10eaxistrue, log10eaxis=log10eaxis, 
          longitudeaxistrue=longitudeaxistrue, longitudeaxis=longitudeaxis, latitudeaxistrue=latitudeaxistrue, latitudeaxis=latitudeaxis,
          logjacob=logjacob, save_directory = resources_dir, logpsf=psf_test, logedisp=edisp_test, aeff=aefffunc,
          pointsources=True, save_results=True, outputresults=False):
    """Produces default IRF normalisation matrices and default astrophysical flux matrix

    Args:
        setup_irfnormalisations (bool, optional): Bool to where a True value 
            would setup the normalisations for the input CTA irfs for the given
            measured and true value axes. 
            Defaults to 1.

        setup_astrobkg (bool, optional): Bool to where a True value 
            would setup the astrophysical flux value for the CTA for the given
            measured and true value axes. Defaults to 1.

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

        aeff (func, optional): _description_. Defaults to aefffunc.

        pointsources (bool, optional): _description_. Defaults to True.

        save_results (bool, optional): _description_. Defaults to True.

        outputresults (bool, optional): _description_. Defaults to False.
    """
    def powerlaw(energy, index, phi0=1):
        """_summary_

        Args:
            energy (_type_): _description_
            index (_type_): _description_
            phi0 (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        return phi0*energy**(index)


    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


    if setup_irfnormalisations:
        logpsfnorm = []
        for logeval in tqdm(log10eaxistrue, desc='Setting up psf normalisation', ncols=80):
            log_psflogerow = []
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

                # psfvals = psf(rad, log10eaxistrue_mesh.flatten(), offset).reshape(log10eaxistrue_mesh.shape)
                logpsfvals = logpsf(longitudeaxis_mesh.flatten(), 
                              latitudeaxis_mesh.flatten(), 
                              log10eaxistrue_mesh.flatten(), 
                              longitudeaxistrue_mesh.flatten(), 
                              latitudeaxistrue_mesh.flatten()).reshape(log10eaxistrue_mesh.shape)
                
                logpsfnormvals = special.logsumexp(logpsfvals, axis=(-2,-1))
                
                log_psflogerow.append(logpsfnormvals)
            logpsfnorm.append(log_psflogerow)
                

    # 
        logpsfnorm = np.squeeze(np.array(logpsfnorm))



        logedispnorm = []
        for logeval in tqdm(log10eaxistrue, desc='Setting up edisp normalisation', ncols=80):
            log10eaxistrue_mesh, longitudeaxistrue_mesh, latitudeaxistrue_mesh, log10eaxis_mesh  = np.meshgrid(logeval,
                                                                                                                longitudeaxistrue, 
                                                                                                                latitudeaxistrue, 
                                                                                                                log10eaxis,
                                                                                                                indexing='ij')

            # truecoords = np.array([longitudeaxistrue_mesh.flatten(), latitudeaxistrue_mesh.flatten()])
            
            # offset = convertlonlat_to_offset(truecoords)

            # edispvals = np.squeeze(edisp(log10eaxis_mesh.flatten(), log10eaxistrue_mesh.flatten(), offset).reshape(log10eaxistrue_mesh.shape))
            logedispvals = logedisp(log10eaxis_mesh.flatten(), 
                              log10eaxistrue_mesh.flatten(), 
                              longitudeaxistrue_mesh.flatten(), 
                              latitudeaxistrue_mesh.flatten()).reshape(log10eaxistrue_mesh.shape)
                
            logedispnormvals = np.squeeze(special.logsumexp(logedispvals+logjacob, axis=-1))
            
            logedispnorm.append(logedispnormvals)


        logedispnorm = np.array(logedispnorm)

        logedispnorm[np.isneginf(logedispnorm)] = 0
        logpsfnorm[np.isneginf(logpsfnorm)] = 0

        if save_results:
            np.save(save_directory+"/psfnormalisation.npy", logpsfnorm)
            np.save(save_directory+"/edispnormalisation.npy", logedispnorm)

    if setup_astrobkg:
        print("Setting up the astrophysical background\n\n")

        print("Setting up HESS sources\n")
        
        hess_catalog = SourceCatalogHGPS(resources_dir+"/hgps_catalog_v1.fits.gz")

        hess_models = hess_catalog.to_models()

        print(f"\nThere are {len(hess_models)} sources in total within the HGPS.")
        
        trueenergyaxis = 10**log10eaxistrue*u.TeV

        energy_axis_true = MapAxis.from_nodes(trueenergyaxis, interp='log', name="energy_true")

        HESSgeom = WcsGeom.create(
            skydir=SkyCoord(0, 0, unit="deg", frame='galactic'),
            binsz=longitudeaxistrue[1]-longitudeaxistrue[0],
            width=(longitudeaxistrue[-1]-longitudeaxistrue[0]+longitudeaxistrue[1]-longitudeaxistrue[0], latitudeaxistrue[-1]-latitudeaxistrue[0]+longitudeaxistrue[1]-longitudeaxistrue[0]),
            frame="galactic",
            proj="CAR",
            axes=[energy_axis_true],
        )

        HESSmap = Map.from_geom(HESSgeom)
        
        hessenergyaxis = energy_axis_true.center.value 
        
        
        count=0 # To keep track of the number of sources satisfying the conditions

        full_hess_flux = 0 #Extracting the flux values for the sources along the axes previously defined

        for idx, model in enumerate(hess_models):
            
            # We will store the longitude 'l' and latitude 'b' values of the source
            temp_l_value = model.spatial_model.position.l.value
            temp_b_value = model.spatial_model.position.b.value
            
            
            # The unit convention for the longitude works as 357, 358, 359, 0, 1, 2
                # we will make it so that it's -3, -2, -1, 0, 1, 2
            if temp_l_value>180:
                temp_l_value=temp_l_value-360
                
                
            # If the location of the source satisfies the criteria above then we start to store it's flux values
            if np.abs(temp_l_value)<5 and np.abs(temp_b_value)<5:
            
                try:
                    # We extract the flux by assigning it to the HESSmap object
                    HESSmap.quantity = model.evaluate_geom(HESSmap.geom)
                    
                    # We can then extract the actual values by referencing the data of this object
                    startdata = HESSmap.data
                    
                    # Removing nan values, '~' is a shortcut for 'not'
                    data = startdata[~np.isnan(startdata)]
                    
                    # If statement to catch sources that have no obervable flux within the region
                    if data.size!=0:
                        
                        # Making sure the data shape is consistent
                        data = data.reshape(startdata.shape)
                        
                        # Adding the flux from this model to the total flux
                        full_hess_flux+=data
                        
                except:
                    print("Something weird happened")
                    print(idx, '\n\n')
                    
                count+=1
            
        # Transposing the longitude and latitude values such that the order of indices goes [logenergy_index, longitude_index, latitude_index]
        full_hess_flux = np.transpose(full_hess_flux, axes=(0,2,1))
        full_hess_flux*=100**2
        full_hess_flux = np.flip(full_hess_flux, axis=1)
        
        hessgeom = HESSmap.geom

        # Extracting the longitudevalues
        hesslonvals = hessgeom.get_coord().lon.value[0][0]

        # Reversing the order as the order is flipped by convention
        hesslonvals = hesslonvals[::-1]

        # Again converting the longitude axis from 357,358,359,0,1,2,3 to -3,-2,-1,0,1,2,3
        hesslonvals[hesslonvals>180] = hesslonvals[hesslonvals>180]-360


        
    
        
        def powerlaw(energy, index, phi0=1):
            return phi0*energy**(index)
        
        
        template_diffuse_skymap = TemplateSpatialModel.read(
            filename=resources_dir+"/gll_iem_v06_gc.fits.gz", normalize=True
        )


        diffuse_iem = SkyModel(
            spatial_model=template_diffuse_skymap,
            spectral_model=PowerLawSpectralModel(),
            name="diffuse-iem",
        )
        
        fermievaluated = np.flip(np.transpose(diffuse_iem.evaluate_geom(HESSgeom), axes=(0,2,1)), axis=1).to(1/u.TeV/u.s/u.sr/(u.m**2))

        fermi_integral_values= special.logsumexp(np.log(fermievaluated.value.T)+np.log(10**log10eaxistrue)+np.log(np.log(10))+np.log(np.diff(log10eaxistrue)[0]), axis=2).T
        # fermievaluated_normalised = np.log(fermievaluated.value) - fermi_integral_values

        fermi_integral_values = fermi_integral_values - special.logsumexp(fermi_integral_values+np.log(np.diff(longitudeaxistrue)[0]*np.diff(latitudeaxistrue)[0]))

        # Slight change in normalisation due to the use of m^2 not cm^2 so there is a 10^4 change in the normalisation
        fermi_gaggero = np.exp(fermi_integral_values+np.log(powerlaw(10**log10eaxistrue, index=-2.41, phi0=1.36*1e-4))[:, np.newaxis, np.newaxis])

        
        # By default the point sources are considered, but they can be turned off use the 'pointsources' argument either
            # from command line/terminal or through the function itself
        if pointsources:
            combinedplotmap = np.logaddexp(np.log(fermi_gaggero), np.log(full_hess_flux))
        else:
            combinedplotmap = np.log(fermi_gaggero)
        
                
        aeff = irfs['aeff']
        aefffunc = lambda energy, offset: aeff.evaluate(energy_true = energy*u.TeV, offset=offset*u.deg).to(u.m**2).value

        energymesh, lonmesh, latmesh = np.meshgrid(10**log10eaxistrue, longitudeaxistrue, latitudeaxistrue, indexing='ij')
        
        aefftable = aefffunc(energymesh, np.sqrt((lonmesh**2)+(latmesh**2)))
        
        combinedplotmapwithaeff = np.exp(combinedplotmap+np.log(aefftable))
        
        if save_results:
            np.save(save_directory+"/unnormalised_astrophysicalbackground.npy", combinedplotmapwithaeff)
        
            print('''Done setup, results saved to package_data. Accessible through `load_package_defaults`(.py) 
        or through `resource_dir` variable in utils.''')
    if outputresults:
        if setup_astrobkg and setup_irfnormalisations:
            return logpsfnorm, logedispnorm, combinedplotmapwithaeff
        elif setup_irfnormalisations:
            return logpsfnorm, logedispnorm
        elif setup_astrobkg:
            return combinedplotmapwithaeff
        else:
            raise Exception("No outputs selected?")
            



if run_function:
    default_file_setup(setup_astrobkg=setup_astrobkg, setup_irfnormalisations=setup_irfnormalisations)