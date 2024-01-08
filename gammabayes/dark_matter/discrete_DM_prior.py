from gammabayes.priors import discrete_logprior
from os import path
from gammabayes.dark_matter.combine_DM_class import combine_DM_models

import time


class dm_discrete_prior(discrete_logprior):

    def __init__(self, 
                 spectral_class_kwds = {},
                 spatial_class_kwds = {},
                 spectral_params = {}, 
                 spatial_params = {},
                 dm_model = 'Scalar Singlet',
                 params ={}, 
                 *args, **kwargs):
        self.dm_model = combine_DM_models(spectral_class_kwds)
        super().__init__(
            logfunction=self.irf_loglikelihood.single_loglikelihood, 
            *args, **kwargs
        )
