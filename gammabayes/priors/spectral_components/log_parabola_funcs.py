from .base_spectral_comp import BaseSpectral_PriorComp
from astropy import units as u


try:
    from jax import numpy as np
except:
    import numpy as np

from icecream import ic




def log_log_parabola_func(self, energy: float|u.Quantity, index: float=2.5, phi0: float|u.Quantity =None, beta: float|u.Quantity=None, E0TeV=1) -> float|u.Quantity:
    """
    Evaluates the log of a log parabola function.
    """
    if phi0 is None:
        phi0 = self.default_parameter_values['phi0']
    if index is None:
        index = self.default_parameter_values['index']
    if beta is None:
        beta = self.default_parameter_values['beta']
    if E0TeV is None:
        E0TeV = self.default_parameter_values['E0']



    energy_ratio = energy/E0TeV
    exponent = -(index + beta*np.log10(energy_ratio))
    log_value = np.log(phi0)+exponent*np.log(energy_ratio)

    return log_value


# Values from here https://arxiv.org/pdf/1903.06621

class LogParabola(BaseSpectral_PriorComp):

    log_parabola_func = log_log_parabola_func

    def __init__(self, default_parameter_values=None,*args, **kwargs):
        
        if default_parameter_values is None:
            default_parameter_values = {}
            
        default_parameter_values.update({'index':2.5, "phi0": 4.47e-7, 'beta':0.37, 'index':2.39})

        super().__init__(logfunc=self.log_parabola_func, default_parameter_values=default_parameter_values)