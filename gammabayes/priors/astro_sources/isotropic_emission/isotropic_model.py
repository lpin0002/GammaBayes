from gammabayes import resources_dir, haversine
from gammabayes.priors.spectral_components import PowerLaw
from gammabayes.priors.spatial_components import BaseSpatial_PriorComp
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

from gammabayes.priors.core import DiscreteLogPrior, SourceFluxDiscreteLogPrior, TwoCompFluxPrior
from gammabayes import GammaLogExposure, GammaBinning

from gammabayes.priors.spectral_components import BrokenPowerLaw
from gammabayes.priors.spatial_components import IsotropicSpatial_PriorComp




class IsotropicGRB(TwoCompFluxPrior):

    def __init__(self, 
                 axes: list[np.ndarray[u.Quantity]]| tuple[np.ndarray[u.Quantity]]=None,
                 binning_geometry: GammaBinning = None,
                 *args, **kwargs):
        
        self._create_geometry(axes=axes, binning_geometry=binning_geometry)

        super().__init__(
            name='IsotropicGRB',
            spectral_class = BrokenPowerLaw, 
            spatial_class = IsotropicSpatial_PriorComp, 
            axes=self.binning_geometry.axes,
            binning_geometry=self.binning_geometry,
            spectral_class_kwds = {'index':2.3, 'cutoff_energy_TeV':0.3, 'phi0':1e-1*u.Unit("TeV-1 cm-2 s-1 sr-1")},
            *args, **kwargs
        )
        

