import numpy as np
from gammabayes.dark_matter.density_profiles import DM_Profiles
from gammabayes.dark_matter.models.Z2_ScalarSinglet import Z2_ScalarSinglet
from os import path
from gammabayes.dark_matter.density_profiles import DM_Profile
from gammabayes.dark_matter.spectral_models import DM_ContinuousEmission_Spectrum
from gammabayes.likelihoods.irfs import IRF_LogLikelihood
from gammabayes.priors.core import TwoCompPrior
import time



class CombineDMComps(TwoCompPrior):

    def __init__(self, 
                 spectral_class: DM_ContinuousEmission_Spectrum, 
                 spatial_class: DM_Profile, 
                 *args, **kwargs
                 ):
        

        super().__init__(spectral_class=spectral_class, 
                         spatial_class =spatial_class,
                         *args, **kwargs
        )


    def convert_param_to_sigmav(self,):
        pass


    


        

