from gammabayes.utils.utils import power_law, resources_dir, convertlonlat_to_offset, power_law
from gammabayes.utils.event_axes import makelogjacob, longitudeaxistrue, latitudeaxistrue, log10eaxistrue
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


def construct_fermi_gaggero_matrix(log_aeff=log_aeff):

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
    fermi_integral_values= logsumexp(np.log(fermievaluated.value.T)+makelogjacob(log10eaxistrue), axis=2).T
    fermi_integral_values = fermi_integral_values - logsumexp(fermi_integral_values+np.log(np.diff(longitudeaxistrue)[0]*np.diff(latitudeaxistrue)[0]))

    # Slight change in normalisation due to the use of m^2 not cm^2 so there is a 10^4 change in the normalisation
    fermi_gaggero = np.exp(fermi_integral_values+np.log(power_law(10**log10eaxistrue, index=-2.41, phi0=1.36*1e-4))[:, np.newaxis, np.newaxis])

    energymesh, lonmesh, latmesh = np.meshgrid(10**log10eaxistrue, longitudeaxistrue, latitudeaxistrue, indexing='ij')
    
    log_aeff_table = log_aeff(energymesh.flatten(), lonmesh.flatten(), latmesh.flatten()).reshape(energymesh.shape)


    log_diffuse_background_source_flux = np.log(fermi_gaggero)

    diffuse_background_observed_event_rate = np.exp(log_diffuse_background_source_flux+log_aeff_table)

    return diffuse_background_observed_event_rate

def log_fermi_gaggero_bkg(log_aeff=log_aeff, normalise=True, log10eaxis=log10eaxistrue):
    log_fermi_diffuse = np.log(construct_fermi_gaggero_matrix(log_aeff=log_aeff))

    if normalise:
        log_fermi_diffuse = log_fermi_diffuse - logsumexp(log_fermi_diffuse+makelogjacob(log10eaxis)[:, None, None])

    log_fermi_diffuse_interpolator = interpolate.RegularGridInterpolator(
        (log10eaxistrue, longitudeaxistrue, latitudeaxistrue), 
        np.exp(log_fermi_diffuse)
        )

    log_fermi_diffuse_func = lambda logenergy, longitude, latitude: np.log(
        log_fermi_diffuse_interpolator(
            (logenergy, longitude, latitude)
            ))

    return log_fermi_diffuse_func

