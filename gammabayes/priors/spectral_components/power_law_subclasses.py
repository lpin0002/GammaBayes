from .base_spectral_comp import BaseSpectral_PriorComp
from astropy import units as u


try:
    from jax import numpy as np
except:
    import numpy as np





from icecream import ic


def log_power_law(self, energy: float|u.Quantity, index: float=2.5, phi0: int|u.Quantity =None) -> float|u.Quantity:
    """
    Evaluates a power law function.

    Args:
        energy (float | Quantity): Energy values.
        index (float): Power law index. Defaults to 2.5 .
        phi0 (int | Quantity, optional): Normalization constant. Defaults to 1.

    Returns:
        float | Quantity: Computed power law values.
    """
    if phi0 is None:
        phi0 = self.default_parameter_values['phi0']
    if index is None:
        index = self.default_parameter_values['index']


    log_value = np.log(phi0) - index * np.log(energy)

    return log_value

class PowerLaw(BaseSpectral_PriorComp):

    log_power_law = log_power_law

    def __init__(self, default_parameter_values=None, *args, **kwargs):
        
        if default_parameter_values is None:
            default_parameter_values = {'index':2.5, "phi0": 1e-8}

        super().__init__(logfunc=self.log_power_law, default_parameter_values=default_parameter_values)



# Defined outside of class for multiprocessing
# TODO: TechDebt
def log_broken_power_law(self, energy: float|u.Quantity, index: float=2.5, cutoff_energy_TeV:float=1, phi0: int|u.Quantity =None) -> float|u.Quantity:
    """
    Evaluates a broken power law function. 

    $\phi_0 \left(\frac{E}{1\, TeV}\right)^{-index} \; e^{-E/cutoff\_energy\_TeV}$

    Args:
        energy (float | Quantity): Energy values.
        index (float): Power law index. Defaults to 2.5
        cutoff_energy_TeV (float): Cut off energy in TeV. Defaults to 1 (1 TeV)
        phi0 (int | Quantity, optional): Normalization constant. Defaults to 1.
        

    Returns:
        float | Quantity: Computed power law values.
    """
    if phi0 is None:
        phi0 = self.default_parameter_values['phi0']
    if index is None:
        index = self.default_parameter_values['index']
    if cutoff_energy_TeV is None:
        cutoff_energy_TeV = self.default_parameter_values['cutoff_energy_TeV']

    log_value =  np.log(phi0) - index*np.log(energy) - energy/cutoff_energy_TeV

    return log_value


class BrokenPowerLaw(BaseSpectral_PriorComp):

    log_broken_power_law = log_broken_power_law

    def __init__(self, default_parameter_values=None, *args, **kwargs):
        if default_parameter_values is None:
            default_parameter_values = {'index':2.5,"cutoff_energy_TeV":1, "phi0": 1e-8, }

        super().__init__(logfunc=self.log_broken_power_law, default_parameter_values=default_parameter_values)


# Influenced by model in https://arxiv.org/pdf/astro-ph/0607333
def log_broad_broken_power_law(self, energy: float|u.Quantity, cutoff_energy_TeV:float=10,
                               index1: float=2.5, index2:float=3.3, S:float=0.3,
                               phi0: int|u.Quantity =None) -> float|u.Quantity:
    """
    Evaluates a broken power law function. 

    $\frac{dN}{dE} = \Phi_0 (E/E_c)^{-\Gamma_1} \left( 1 + \left(E/E_c\right)^{1/S}\right)^{S(\Gamma_1-\Gamma_2)}$

    Args:
        energy (float | Quantity): Energy values.
        index1 (float): Power law index before break. Defaults to 2.5
        index2 (float): Power law index after break. Defaults to 1
        S (float): Width of the transition region. Defaults to 0.3.
        cutoff_energy_TeV (float): Cut off energy in TeV. Defaults to 10 (10 TeV)
        phi0 (int | Quantity, optional): Normalization constant. Defaults to 1.
        

    Returns:
        float | Quantity: Computed power law values.
    """
    if phi0 is None:
        phi0 = self.default_parameter_values['phi0']
    if S is None:
        S = self.default_parameter_values['S']
    if index1 is None:
        index1 = self.default_parameter_values['index1']
    if index2 is None:
        index2 = self.default_parameter_values['index2']
    if cutoff_energy_TeV is None:
        cutoff_energy_TeV = self.default_parameter_values['cutoff_energy_TeV']

    log_value =  np.log(phi0) - index1*np.log(energy/cutoff_energy_TeV) + S*(index1-index2)*np.log(1+(energy/cutoff_energy_TeV)**(1/S))

    return log_value



class BroadBrokenPowerLaw(BaseSpectral_PriorComp):

    log_broad_broken_power_law = log_broad_broken_power_law

    def __init__(self, default_parameter_values=None, *args, **kwargs):
        if default_parameter_values is None:
            default_parameter_values = {'index1':2.5,'index2':2.5,"S":0.3, "cutoff_energy_TeV":10, "phi0": 1e-8, }

        super().__init__(logfunc=self.log_broad_broken_power_law, default_parameter_values=default_parameter_values)