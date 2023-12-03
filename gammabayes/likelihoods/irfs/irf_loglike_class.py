from gammabayes.likelihoods.core import discrete_loglike
from gammabayes.likelihoods.irfs.irf_extractor_class import irf_extractor
import numpy as np
class irf_loglikelihood(discrete_loglike):

    def __init__(self,pointing_direction=np.asarray([0,0]).T, zenith=20, hemisphere='South', prod_vers=5, *args, **kwargs):
        irf_loglikelihood = irf_extractor(zenith_angle=zenith, hemisphere=hemisphere, prod_vers=prod_vers)
        super().__init__(
            logfunction=irf_loglikelihood.single_loglikelihood, 
            *args, **kwargs
        )
        self.pointing_direction = pointing_direction

    def __call__(self, *args, **kwargs):
        
        return self.logfunction(pointing_direction=self.pointing_direction, *args, **kwargs)

    
