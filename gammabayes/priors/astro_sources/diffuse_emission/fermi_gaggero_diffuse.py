from gammabayes.utils import power_law, resources_dir, power_law, haversine
from gammabayes.utils import iterate_logspace_integration, logspace_riemann
from gammabayes.likelihoods.irfs import IRF_LogLikelihood
import numpy as np
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
    TemplateSpatialModel,
    create_fermi_isotropic_diffuse_model,
)
from gammapy.catalog import SourceCatalogHGPS
from gammapy.maps import Map, MapAxis, WcsGeom

from scipy import interpolate
from scipy.special import logsumexp
from astropy.coordinates import SkyCoord
from astropy import units as u

from gammabayes.priors.core import DiscreteLogPrior


def construct_fermi_gaggero_matrix(energy_axis: np.ndarray, longitudeaxis: np.ndarray, latitudeaxis: np.ndarray,
                                   log_aeff: callable, logspace_integrator:callable=logspace_riemann):
    """Constructs the Fermi-Gaggero diffuse background event rate matrix (using Gammapy).

    Args:
        energy_axis (np.ndarray): Energy axis for the matrix (TeV).
        longitudeaxis (np.ndarray): Longitude axis for the matrix (degrees).
        latitudeaxis (np.ndarray): Latitude axis for the matrix (degrees).
        log_aeff (callable): Logarithm of the effective area as a function of energy, longitude, and latitude.
        logspace_integrator (callable, optional): Function used for integration over log space. Defaults to logspace_riemann.

    Returns:
        np.ndarray: A 3D array representing the observed event rate from the Fermi-Gaggero diffuse background over specified axes.
    """

    energy_axis_true = MapAxis.from_nodes(energy_axis*u.TeV, interp='log', name="energy_true")

    HESSgeom = WcsGeom.create(
        skydir=SkyCoord(0, 0, unit="deg", frame='galactic'),
        binsz=(np.diff(longitudeaxis)[0], np.diff(latitudeaxis)[0]),
        width=(np.ptp(longitudeaxis)+np.diff(longitudeaxis)[0], np.ptp(latitudeaxis)+np.diff(latitudeaxis)[0]),
        frame="galactic",
        proj="CAR",
        axes=[energy_axis_true],
    )

    template_diffuse_skymap = TemplateSpatialModel.read(
        filename=resources_dir+"/gll_iem_v06_gc.fits.gz", normalize=True
    )

    diffuse_iem = SkyModel(
        spatial_model=template_diffuse_skymap,
        spectral_model=PowerLawSpectralModel(),
        name="diffuse-iem",
    )

    # Need to flip as convention goes positive to negative 
    fermievaluated = np.flip(np.transpose(diffuse_iem.evaluate_geom(HESSgeom), axes=(0,2,1)), axis=1).to(1/u.TeV/u.s/u.sr/(u.m**2))

    # Normalising so I can apply the normalisation of that in Gaggero et al.
    fermi_integral_values= logspace_integrator(logy=np.log(fermievaluated.value), x=energy_axis, axis=0)
    fermi_integral_values = fermi_integral_values - logspace_integrator(
        logy=logspace_integrator(
            logy=fermi_integral_values, x=longitudeaxis, axis=0), x=latitudeaxis, axis=0)

    # Slight change in normalisation due to the use of m^2 not cm^2 so there is a 10^4 change in the normalisation
    fermi_gaggero = np.exp(fermi_integral_values+np.log(power_law(energy_axis, index=-2.41, phi0=1.36*1e-8))[:, np.newaxis, np.newaxis])

    energymesh, lonmesh, latmesh = np.meshgrid(energy_axis, longitudeaxis, latitudeaxis, indexing='ij')
    
    log_aeff_table = log_aeff(energymesh.flatten(), lonmesh.flatten(), latmesh.flatten()).reshape(energymesh.shape)


    log_diffuse_background_source_flux = np.log(fermi_gaggero)

    diffuse_background_observed_event_rate = np.exp(log_diffuse_background_source_flux+log_aeff_table)

    return diffuse_background_observed_event_rate

class construct_log_fermi_gaggero_bkg(object):
    """        
    Args:
        energy_axis (np.ndarray): Energy axis for the model (TeV).
        
        longitudeaxis (np.ndarray): Longitude axis for the model (degrees).
        
        latitudeaxis (np.ndarray): Latitude axis for the model (degrees).
        
        log_aeff (callable): Logarithm of the effective area as a function of energy, longitude, and latitude.
        
        normalise (bool, optional): Whether to normalize the model over the specified axes. Defaults to True.
        
        logspace_integrator (callable, optional): Function used for normalization in log space. Defaults to logspace_riemann.

    This class generates a regular grid interpolator for the Fermi-Gaggero diffuse background model,
    allowing for log-space evaluation of the model at arbitrary points.
    """

    def __init__(self, energy_axis: np.ndarray, longitudeaxis: np.ndarray, latitudeaxis: np.ndarray,
    log_aeff: callable, normalise: bool=True, logspace_integrator: callable=logspace_riemann):
        """Constructs and interpolates the log of the Fermi-Gaggero diffuse background model.

        Args:
            energy_axis (np.ndarray): Energy axis for the model (TeV).
            
            longitudeaxis (np.ndarray): Longitude axis for the model (degrees).
            
            latitudeaxis (np.ndarray): Latitude axis for the model (degrees).
            
            log_aeff (callable): Logarithm of the effective area as a function of energy, longitude, and latitude.
            
            normalise (bool, optional): Whether to normalize the model over the specified axes. Defaults to True.
            
            logspace_integrator (callable, optional): Function used for normalization in log space. Defaults to logspace_riemann.

        This class generates a regular grid interpolator for the Fermi-Gaggero diffuse background model,
        allowing for log-space evaluation of the model at arbitrary points.
        """
    
        axes = [energy_axis, longitudeaxis, latitudeaxis]
        log_fermi_diffuse = np.log(construct_fermi_gaggero_matrix(energy_axis=energy_axis, 
        longitudeaxis=longitudeaxis, latitudeaxis=latitudeaxis, log_aeff=log_aeff))

        if normalise:
            log_fermi_diffuse = log_fermi_diffuse - logspace_integrator(
                logy = logspace_integrator(
                    logy =logspace_integrator(
                        logy=log_fermi_diffuse, 
                        x=energy_axis, axis=0), 
                    x=longitudeaxis, axis=0), 
                x=latitudeaxis, axis=0)

        # Have to interpolate actual probabilities as otherwise these maps include -inf
        self.fermi_diffuse_interpolator = interpolate.RegularGridInterpolator(
            (*axes,), 
            np.exp(log_fermi_diffuse) 
            )

    # Then we make a wrapper to put the result of the function in log space
    def log_func(self, energy, longitude, latitude, 
                 spectral_parameters={}, spatial_parameters={}): 
        """Computes the log of the interpolated Fermi-Gaggero model at given energy, longitude, and latitude.

        Args:
            energy (float): Energy value for evaluation (TeV).
            longitude (float): Longitude value for evaluation (degrees).
            latitude (float): Latitude value for evaluation (degrees).
            spectral_parameters (dict, optional): Spectral parameters, unused in this context. Defaults to an empty dict.
            spatial_parameters (dict, optional): Spatial parameters, unused in this context. Defaults to an empty dict.

        Returns:
            float: The log of the Fermi-Gaggero diffuse background model at the specified location.
        """
        return np.log(
        self.fermi_diffuse_interpolator(
            (energy, longitude, latitude)
            ))
    

# Just a little helper class
class FermiGaggeroDiffusePrior(DiscreteLogPrior):
    """A subclass of DiscreteLogPrior for modeling the Fermi-Gaggero diffuse astrophysical background prior.

    Args:
        energy_axis (np.ndarray): The energy axis for the prior (TeV).
        
        longitudeaxis (np.ndarray): The longitude axis for the prior (degrees).
        
        latitudeaxis (np.ndarray): The latitude axis for the prior (degrees).
        
        irf (IRF_LogLikelihood): Instrument Response Function log likelihood instance, providing `log_aeff`.
        
        normalise (bool, optional): Whether to normalize the prior. Defaults to True.
        
        logspace_integrator (callable, optional): Integrator function for normalization in log space. Defaults to logspace_riemann.

    This class constructs a prior based on the Fermi-Gaggero diffuse background model, incorporating instrument
    response functions and allowing for normalization over the specified energy, longitude, and latitude axes.
    """

    def __init__(self, energy_axis:np.ndarray, longitudeaxis:np.ndarray, latitudeaxis:np.ndarray,  
                 irf: IRF_LogLikelihood, 
                 normalise: bool=True, 
                 logspace_integrator: callable = logspace_riemann, 
                 *args, **kwargs):
        """Initializes an instance of FermiGaggeroDiffusePrior, a subclass of DiscreteLogPrior, 
        designed for modeling the Fermi-Gaggero diffuse astrophysical background prior with respect 
        to given energy, longitude, and latitude axes.

        Args:
            energy_axis (np.ndarray): The energy axis for which the prior is defined, typically in GeV.
            longitudeaxis (np.ndarray): The longitude axis over which the prior is defined, in degrees.
            latitudeaxis (np.ndarray): The latitude axis over which the prior is defined, in degrees.
            irf (IRF_LogLikelihood): An instance of IRF_LogLikelihood, providing the instrument response function, 
                                     specifically `log_aeff`, the log of the effective area.
            normalise (bool, optional): If True, the log prior will be normalized over the specified axes. 
                                        Defaults to True.
            logspace_integrator (callable, optional): A function used for normalization of the log prior in log space. 
                                                      Defaults to logspace_riemann.

        This class utilizes the Fermi-Gaggero model for the diffuse astrophysical gamma-ray background, 
        incorporating the effects of the instrument's effective area as described by the provided IRF.
        """


        self.construct_log_fermi_gaggero_bkg_func_class = construct_log_fermi_gaggero_bkg(
            energy_axis=energy_axis, 
            longitudeaxis=longitudeaxis, 
            latitudeaxis=latitudeaxis,  
            log_aeff= irf.log_aeff, 
            normalise=normalise, 
            logspace_integrator=logspace_integrator
        )
        super().__init__(
            name='Fermi-Gaggero Diffuse Astrophysical Prior',
            axes_names=['energy', 'lon', 'lat'],
            axes=(energy_axis, longitudeaxis, latitudeaxis),
            logfunction=self.construct_log_fermi_gaggero_bkg_func_class.log_func, 
            *args, **kwargs
        )
        






