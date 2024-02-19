import numpy as np
import astropy.units as u


import numpy as np
import astropy.units as u
from gammabayes.utils import logspace_riemann, haversine, update_with_defaults
from .base_dm_profile import DM_Profile
import time
from gammapy.astro.darkmatter.profiles import (
    DMProfile as Gammapy_DMProfile,
    EinastoProfile as Gammapy_EinastoProfile,
    NFWProfile as Gammapy_NFWProfile,
    BurkertProfile as Gammapy_BurkertProfile,
    MooreProfile as Gammapy_MooreProfile,
    IsothermalProfile as Gammapy_IsothermalProfile,
)


class Einasto_Profile(DM_Profile):

    @staticmethod
    def log_profile(radius: float | np.ndarray , 
                    r_s: float, 
                    alpha: float, 
                    rho_s: float) -> float | np.ndarray :
        rr = radius / r_s
        exponent = (2 / alpha) * (rr**alpha - 1)
        return np.log(rho_s) + -1 * exponent

    def __init__(self, 
                 default_alpha = 0.17, 
                 default_rho_s: float = 1., 
                 default_r_s: float = 28.4, 
                 *args, **kwargs):
        super().__init__(
            log_profile_func=self.log_profile, 
            kwd_profile_default_vals = {'r_s': default_r_s, 
                                        'alpha': default_alpha, 
                                        'rho_s':default_rho_s},
                                        gammapy_profile_class=Gammapy_EinastoProfile,
            *args, **kwargs
        )



class GNFW_Profile(DM_Profile):

    @staticmethod
    def log_profile(radius: float | np.ndarray , 
                         r_s: float, 
                         alpha: float, 
                         beta: float, 
                         gamma: float, 
                         rho_s: float):
        rr = radius / r_s
        return np.log(rho_s) -  (
            gamma*np.log(rr) + ((beta - gamma) / alpha) * np.log(1 + rr **  alpha)
        )
    
    def __init__(self, 
                 default_alpha: float | int = 1, 
                 default_beta: float | int = 3, 
                 default_gamma: float | int =1, 
                 default_rho_s: float = 1., 
                 default_r_s: float = 24.42, *args, **kwargs):
        super().__init__(
            log_profile_func=self.log_profile, 
            kwd_profile_default_vals = {'r_s': default_r_s, 
                                        'alpha': default_alpha, 
                                        'beta': default_beta,
                                        'gamma': default_gamma,
                                        'rho_s':default_rho_s},
            *args, **kwargs
        )



class Burkert_Profile(DM_Profile):

    @staticmethod
    def log_profile(radius, r_s, rho_s):
        rr = radius / r_s
        return np.log(rho_s) - (np.log(1 + rr) + np.log(1 + rr**2))
    
    def __init__(self, 
                 default_rho_s: float = 1., 
                 default_r_s: float = 12.67, *args, **kwargs):
        super().__init__(
            log_profile_func=self.log_profile, 
            kwd_profile_default_vals = {'r_s': default_r_s, 
                                        'rho_s':default_rho_s},
            *args, **kwargs
        )
 


class Moore_Profile(DM_Profile):

    @staticmethod
    def log_profile(radius, r_s, rho_s):
        rr = radius / r_s
        return np.log(rho_s) - 1.16*np.log(rr)  - 1.84* np.log(1 + rr)
    
    def __init__(self, 
                 default_rho_s: float = 1., 
                 default_r_s: float = 30.28, *args, **kwargs):
        super().__init__(
            log_profile_func=self.log_profile, 
            kwd_profile_default_vals = {'r_s': default_r_s, 
                                        'rho_s':default_rho_s},
            *args, **kwargs
        )
