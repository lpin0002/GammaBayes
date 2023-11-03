# If running from main file, the terminal format should be $ python -m gammabayes.utils.default_file_setup 1 1 1
    # If you're running from a script there shouldn't be any issues as setup is just a func
from ..utils.event_axes import log10eaxistrue, longitudeaxistrue, latitudeaxistrue, log10eaxis, longitudeaxis, latitudeaxis, logjacob
from ..utils.utils import angularseparation, convertlonlat_to_offset, resource_dir
from tqdm import tqdm

from ..likelihoods.instrument_response_funcs import irfs, log_edisp, log_psf

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



def default_file_setup(setup_irfnormalisations=1, setup_astrobkg=1, log10eaxistrue=log10eaxistrue, log10eaxis=log10eaxis, 
          longitudeaxistrue=longitudeaxistrue, longitudeaxis=longitudeaxis, latitudeaxistrue=latitudeaxistrue, latitudeaxis=latitudeaxis,
          logjacob=logjacob, save_directory = resource_dir, psf=log_psf, edisp=log_edisp, aeff=aefffunc,
          pointsources=True, 
          save_results=True, outputresults=False, out_individual_astro_backgrounds=0):
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
    
    print(f"Save directory is {save_directory}")

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


    if setup_irfnormalisations:
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

    if setup_astrobkg:
        print("Setting up the astrophysical background\n\n")

        print("Setting up HESS sources\n")
        
        hess_catalog = SourceCatalogHGPS(resource_dir+"/hgps_catalog_v1.fits.gz")

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


        
        
        template_diffuse_skymap = TemplateSpatialModel.read(
            filename=resource_dir+"/gll_iem_v06_gc.fits.gz", normalize=True
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

        
       
        
                
        aeff = irfs['aeff']
        aefffunc = lambda energy, offset: aeff.evaluate(energy_true = energy*u.TeV, offset=offset*u.deg).to(u.m**2).value

        energymesh, lonmesh, latmesh = np.meshgrid(10**log10eaxistrue, longitudeaxistrue, latitudeaxistrue, indexing='ij')
        
        aefftable = aefffunc(energymesh, np.sqrt((lonmesh**2)+(latmesh**2)))

         # By default the point sources are considered, but they can be turned off use the 'pointsources' argument either
            # from command line/terminal or through the function itself
        if pointsources:
            print("Including point sources.")
            if out_individual_astro_backgrounds:
                print("Outputting individual astro sources")

                log_diffuse_background_source_flux = np.log(fermi_gaggero)
                log_point_hess_background_source_flux = np.log(full_hess_flux)

                diffuse_background_event_rate = np.exp(log_diffuse_background_source_flux+np.log(aefftable))
                point_hess_background_event_rate = np.exp(log_point_hess_background_source_flux+np.log(aefftable))
                if save_results:
                    np.save(save_directory+"/unnormalised_astrophysical_diffuse_background.npy", diffuse_background_event_rate)
                    np.save(save_directory+"/unnormalised_astrophysical_point_background.npy", point_hess_background_event_rate)

                
            combinedplotmap = np.logaddexp(np.log(fermi_gaggero), np.log(full_hess_flux))
            combinedplotmapwithaeff = np.exp(combinedplotmap+np.log(aefftable))
            if save_results:
                np.save(save_directory+"/unnormalised_astrophysicalbackground.npy", combinedplotmapwithaeff)
                

        else:
            print("Not including point sources.")
            if out_individual_astro_backgrounds:
                log_diffuse_background_source_flux = np.log(fermi_gaggero)

                diffuse_background_event_rate = np.exp(log_diffuse_background_source_flux+np.log(aefftable))
                if save_results:
                    np.save(save_directory+"/unnormalised_astrophysical_diffuse_background.npy", diffuse_background_event_rate)

            combinedplotmap = np.log(fermi_gaggero)
            combinedplotmapwithaeff = np.exp(combinedplotmap+np.log(aefftable))
            if save_results:
                np.save(save_directory+"/unnormalised_astrophysicalbackground.npy", combinedplotmapwithaeff)
        
        print('''Done setup, results saved to package_data. Accessible through `load_package_data`(.py) 
    or through `resource_dir` variable in utils.''')
    if outputresults:
        if setup_astrobkg and setup_irfnormalisations and not(out_individual_astro_backgrounds):
            return psfnorm, edispnorm, combinedplotmapwithaeff
        elif setup_astrobkg and out_individual_astro_backgrounds and setup_irfnormalisations:
            return psfnorm, edispnorm, combinedplotmapwithaeff, diffuse_background_event_rate, point_hess_background_event_rate
        elif setup_astrobkg and not(out_individual_astro_backgrounds) and not(setup_irfnormalisations):
            return combinedplotmapwithaeff
        elif setup_astrobkg and out_individual_astro_backgrounds and not(setup_irfnormalisations):
            return combinedplotmapwithaeff, diffuse_background_event_rate, point_hess_background_event_rate
        elif setup_irfnormalisations and not(setup_astrobkg):
            return psfnorm, edispnorm
        else:
            raise Exception("You have specified to return results but also to not create any. Please fix.")
            



if __name__=="__main__":
        
    try:
        setup_irfnormalisations = int(sys.argv[1])
    except:
        setup_irfnormalisations = 1
        
    try:
        setup_astrobkg = int(sys.argv[2])
    except:
        setup_astrobkg = 1

    try:
        out_individual_astro_backgrounds = int(sys.argv[3])
    except:
        out_individual_astro_backgrounds = 0

    try:
        save_directory = sys.argv[4]
    except:
        save_directory = 0

    if save_directory!=0:
        print(f"Chosen save directory is: {save_directory}")
        default_file_setup(setup_astrobkg=setup_astrobkg, 
        setup_irfnormalisations=setup_irfnormalisations, 
        out_individual_astro_backgrounds=out_individual_astro_backgrounds,
        save_directory=save_directory)
    else:
        print("No save directory specified.")

        default_file_setup(setup_astrobkg=setup_astrobkg, 
        setup_irfnormalisations=setup_irfnormalisations, 
        out_individual_astro_backgrounds=out_individual_astro_backgrounds)