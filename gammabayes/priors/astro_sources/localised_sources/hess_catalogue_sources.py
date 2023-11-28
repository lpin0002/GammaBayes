
from gammabayes.utils.event_axes import energy_true_axis, longitudeaxistrue, latitudeaxistrue
from gammabayes.utils import resources_dir, convertlonlat_to_offset, iterate_logspace_integration
from gammabayes.likelihoods.irfs.gammapy_wrappers import log_aeff
import numpy as np
from scipy import interpolate
from scipy.special import logsumexp
from astropy.coordinates import SkyCoord
from astropy import units as u
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
)
from gammapy.catalog import SourceCatalogHGPS


def construct_hess_source_map(energy_axis=energy_true_axis, 
    longitudeaxis=longitudeaxistrue, latitudeaxis=latitudeaxistrue,
    log_aeff=log_aeff):

    hess_catalog = SourceCatalogHGPS(resources_dir+"/hgps_catalog_v1.fits.gz")

    hess_models = hess_catalog.to_models()
    
    trueenergyaxis = energy_axis*u.TeV

    energy_axis_true_nodes = MapAxis.from_nodes(trueenergyaxis, interp='log', name="energy_true")

    HESSgeom = WcsGeom.create(
        skydir=SkyCoord(0, 0, unit="deg", frame='galactic'),
        binsz=(np.diff(longitudeaxis)[0], np.diff(latitudeaxis)[0],),
        width=(np.ptp(longitudeaxis)+np.diff(longitudeaxis)[0], np.ptp(latitudeaxis)+np.diff(latitudeaxis)[0]),
        frame="galactic",
        proj="CAR",
        axes=[energy_axis_true_nodes],
    )

    HESSmap = Map.from_geom(HESSgeom)
    
    hessenergyaxis = energy_axis_true_nodes.center.value 
    
    
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
        if np.abs(temp_l_value)<np.max(longitudeaxis) and np.abs(temp_b_value)<np.max(latitudeaxis):
        
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


    energymesh, lonmesh, latmesh = np.meshgrid(energy_axis, longitudeaxis, latitudeaxis, indexing='ij')
    
    log_aeff_table = log_aeff(energymesh.flatten(), lonmesh.flatten(), latmesh.flatten()).reshape(energymesh.shape)

    log_point_hess_background_source_flux = np.log(full_hess_flux)

    point_hess_background_event_rate = np.exp(log_point_hess_background_source_flux+log_aeff_table)

    return point_hess_background_event_rate
                


def construct_hess_source_map_interpolation(energy_true_axis=energy_true_axis, 
    longitudeaxistrue=longitudeaxistrue, latitudeaxistrue=latitudeaxistrue,
    log_aeff=log_aeff, normalise=True, iterate_logspace_integrator=iterate_logspace_integration):
    axes = [energy_true_axis, longitudeaxistrue, latitudeaxistrue]


    log_astro_sourcemap = np.log(construct_hess_source_map(energy_axis=energy_true_axis, 
        longitudeaxis=longitudeaxistrue, latitudeaxis=latitudeaxistrue,
        log_aeff=log_aeff))

    if normalise:
        log_astro_sourcemap = log_astro_sourcemap - iterate_logspace_integrator(log_astro_sourcemap, axes=axes)

    # Have to interpolate actual probabilities as otherwise these maps include -inf
    hess_grid_interpolator = interpolate.RegularGridInterpolator(
        (*axes,), 
        np.exp(log_astro_sourcemap))

    # Then we make a wrapper to put the result of the function in log space
    log_astro_func = lambda energy, longitude, latitude: np.log(hess_grid_interpolator((energy, longitude, latitude)))


    return log_astro_func