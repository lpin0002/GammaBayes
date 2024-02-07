# Necessary imports for the script functionality
from gammabayes.utils import resources_dir, iterate_logspace_integration, haversine
from gammabayes.likelihoods.irfs import IRF_LogLikelihood
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
from gammabayes.priors.core import DiscreteLogPrior


def construct_hess_source_map(energy_axis: np.ndarray, longitudeaxis: np.ndarray, latitudeaxis: np.ndarray,
    log_aeff: callable):
    """Constructs a map of HESS source fluxes based on the HGPS catalog.

    Args:
        energy_axis (np.ndarray): Energy axis for the map (TeV).
        
        longitudeaxis (np.ndarray): Longitude axis for the map (degrees).
        
        latitudeaxis (np.ndarray): Latitude axis for the map (degrees).
        
        log_aeff (callable): Logarithm of the effective area as a function of energy, longitude, and latitude.

    Returns:
        np.ndarray: A 3D array representing the event rate from HESS catalog sources over the specified energy,
                    longitude, and latitude axes.
    """

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
    # log_aeff_table = 0

    log_point_hess_background_source_flux = np.log(full_hess_flux)

    point_hess_background_event_rate = np.exp(log_point_hess_background_source_flux+log_aeff_table)

    return point_hess_background_event_rate
                


class construct_hess_source_map_interpolation(object):
    
    def __init__(self, energy_axis: np.ndarray, longitudeaxis: np.ndarray, latitudeaxis: np.ndarray,
                 log_aeff: callable, normalise: bool = True, iterate_logspace_integrator: callable =iterate_logspace_integration):
        """Provides interpolated values from the HESS source map for given energy, longitude, and latitude points.

        Args:
            energy_axis (np.ndarray): Energy axis for the interpolation (TeV).
            
            longitudeaxis (np.ndarray): Longitude axis for the interpolation (degrees).
            
            latitudeaxis (np.ndarray): Latitude axis for the interpolation (degrees).
            
            log_aeff (callable): Function to calculate the log of the effective area.
            
            normalise (bool, optional): Whether to normalise the source map. Defaults to True.
            
            iterate_logspace_integrator (callable, optional): Integrator function for normalisation in log space. Defaults to iterate_logspace_integration.

        This class constructs an interpolation function for the HESS source map, allowing for the evaluation of the
        map at arbitrary points within the defined axes.
        """
        
        axes = [energy_axis, longitudeaxis, latitudeaxis]


        log_astro_sourcemap = np.log(construct_hess_source_map(energy_axis=energy_axis, 
            longitudeaxis=longitudeaxis, latitudeaxis=latitudeaxis,
            log_aeff=log_aeff))

        if normalise:
            log_astro_sourcemap = log_astro_sourcemap - iterate_logspace_integrator(log_astro_sourcemap, axes=axes)

        # Have to interpolate actual probabilities as otherwise these maps include -inf
        self.hess_grid_interpolator = interpolate.RegularGridInterpolator(
            (*axes,), 
            np.exp(log_astro_sourcemap))

    # Then we make a wrapper to put the result of the function in log space
    def log_func(self, energy, longitude, latitude, 
                       spectral_parameters={}, spatial_parameters={}):
        """Computes the log of interpolated values from the HESS source map at given points.

        Args:
            energy (float): Energy value for the interpolation (TeV).
            
            longitude (float): Longitude value for the interpolation (degrees).
            
            latitude (float): Latitude value for the interpolation (degrees).
            
            spectral_parameters (dict, optional): Spectral parameters for the model. Defaults to an empty dict.
            
            spatial_parameters (dict, optional): Spatial parameters for the model. Defaults to an empty dict.

        Returns:
            float: Log of the interpolated value from the HESS source map at the specified point.
        """
        return np.log(self.hess_grid_interpolator((energy, longitude, latitude)))


class HESSCatalogueSources_Prior(DiscreteLogPrior):
    
    def __init__(self, energy_axis: np.ndarray, longitudeaxis: np.ndarray, latitudeaxis: np.ndarray,
                 irf: IRF_LogLikelihood, 
                 normalise: bool = True, iterate_logspace_integrator: callable =iterate_logspace_integration,
                 *args, **kwargs):
        """Defines a prior based on HESS catalogue sources over specified energy, longitude, and latitude axes.

        Args:
            energy_axis (np.ndarray): Energy axis for the prior (TeV).
            
            longitudeaxis (np.ndarray): Longitude axis for the prior (degrees).
            
            latitudeaxis (np.ndarray): Latitude axis for the prior (degrees).
            
            irf (IRF_LogLikelihood): Instrument Response Function log likelihood instance.
            
            normalise (bool, optional): Whether to normalise the prior. Defaults to True.
            
            iterate_logspace_integrator (callable, optional): Function for integration over log space for normalisation. Defaults to iterate_logspace_integration.
        
        This class utilizes an interpolation of the HESS source map to define a prior distribution for Bayesian analysis,
        integrating the map's flux values with the instrument's effective area.
        """
        

        self.log_hess_class_instance = construct_hess_source_map_interpolation(energy_axis=energy_axis, 
                                                                            longitudeaxis=longitudeaxis, 
                                                                            latitudeaxis=latitudeaxis, 
                                                                            log_aeff=irf.log_aeff,)
        
        super().__init__(
            name='HESS Catalogue Sources Prior',
            axes_names=['energy', 'lon', 'lat'],
            axes=(energy_axis, longitudeaxis, latitudeaxis),
            logfunction=self.log_hess_class_instance.log_func, 
            *args, **kwargs
        )

