from gammabayes.likelihoods.core import discrete_loglike
from gammabayes.likelihoods.irfs.prod5 import single_loglikelihood
import numpy as np
class irf_loglikelihood(discrete_loglike):




    def __init__(self,pointing_direction=np.asarray([0,0]).T, zenith=20, hemisphere='South', *args, **kwargs):
        super().__init__(
            logfunction=single_loglikelihood, 
            *args, **kwargs
        )
        self.pointing_direction = pointing_direction

    def __call__(self, *args, **kwargs):
        
        return self.logfunction(pointing_direction=self.pointing_direction, *args, **kwargs)

    
