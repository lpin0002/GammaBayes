from gammabayes.utils import power_law, resources_dir, convertlonlat_to_offset, power_law
from gammabayes.utils import iterate_logspace_integration, logspace_riemann
from gammabayes.utils.event_axes import longitudeaxistrue, latitudeaxistrue, energy_true_axis
from gammabayes.likelihoods.irfs.gammapy_wrappers import log_aeff
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


def construct_fermi_gaggero_matrix(energy_axis=energy_true_axis, 
    longitudeaxis=longitudeaxistrue, latitudeaxis=latitudeaxistrue, log_aeff=log_aeff, logspace_integrator=logspace_riemann):

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
    fermi_gaggero = np.exp(fermi_integral_values+np.log(power_law(energy_axis, index=-2.41, phi0=1.36*1e-4))[:, np.newaxis, np.newaxis])

    energymesh, lonmesh, latmesh = np.meshgrid(energy_axis, longitudeaxis, latitudeaxis, indexing='ij')
    
    log_aeff_table = log_aeff(energymesh.flatten(), lonmesh.flatten(), latmesh.flatten()).reshape(energymesh.shape)


    log_diffuse_background_source_flux = np.log(fermi_gaggero)

    diffuse_background_observed_event_rate = np.exp(log_diffuse_background_source_flux+log_aeff_table)

    return diffuse_background_observed_event_rate

def construct_log_fermi_gaggero_bkg(energy_axis=energy_true_axis, 
                            longitudeaxis=longitudeaxistrue, 
                            latitudeaxis=latitudeaxistrue, 
                            log_aeff=log_aeff, normalise=True, logspace_integrator=logspace_riemann):
    axes = [energy_axis, longitudeaxis, latitudeaxis]
    log_fermi_diffuse = np.log(construct_fermi_gaggero_matrix(log_aeff=log_aeff))

    if normalise:
        log_fermi_diffuse = log_fermi_diffuse - logspace_integrator(
            logy = logspace_integrator(
                logy =logspace_integrator(
                    logy=log_fermi_diffuse, 
                    x=energy_axis, axis=0), 
                x=longitudeaxis, axis=0), 
            x=latitudeaxis, axis=0)

    # Have to interpolate actual probabilities as otherwise these maps include -inf
    fermi_diffuse_interpolator = interpolate.RegularGridInterpolator(
        (*axes,), 
        np.exp(log_fermi_diffuse) 
        )

    # Then we make a wrapper to put the result of the function in log space
    log_fermi_diffuse_func = lambda energy, longitude, latitude: np.log(
        fermi_diffuse_interpolator(
            (energy, longitude, latitude)
            ))

    return log_fermi_diffuse_func

