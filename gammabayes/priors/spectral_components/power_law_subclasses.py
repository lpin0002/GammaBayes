from .base_spectral_comp import BaseSpectral_PriorComp
from astropy import units as u
import numpy as np

from icecream import ic

class PowerLaw(BaseSpectral_PriorComp):

    
    def log_power_law(self, energy: float|u.Quantity, index: float=2.5, phi0: int|u.Quantity =None) -> float|u.Quantity:
        """
        Evaluates a power law function.

        Args:
            energy (float | Quantity): Energy values.
            index (float): Power law index.
            phi0 (int | Quantity, optional): Normalization constant. Defaults to 1.

        Returns:
            float | Quantity: Computed power law values.
        """
        if phi0 is None:
            phi0 = self.default_parameter_values['phi0']
        if index is None:
            index = self.default_parameter_values['index']

        log_value = np.log(phi0) - index * np.log(energy.to("TeV").value)

        return log_value

    def __init__(self, default_parameter_values=None, energy_units=u.Unit("TeV"), *args, **kwargs):
        if default_parameter_values is None:
            default_parameter_values = {'index':2.5, "phi0": 1e-8}

        super().__init__(logfunc=self.log_power_law, default_parameter_values=default_parameter_values)



class BrokenPowerLaw(BaseSpectral_PriorComp):

    # TODO: TechDebt
    def log_broken_power_law(self, energy: float|u.Quantity, index: float=2.5, cutoff_energy_TeV:float=1, phi0: int|u.Quantity =None) -> float|u.Quantity:
        """
        Evaluates a broken power law function. 

        $\phi_0 \left(\frac{E}{1\, TeV}\right)^{-index} \; e^{-E/cutoff\_energy\_TeV}$

        Args:
            energy (float | Quantity): Energy values.
            index (float): Power law index. Defaults to -2.5
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

        log_value =  np.log(phi0) - index*np.log(energy.to("TeV").value) - energy.to("TeV").value/cutoff_energy_TeV

        return log_value

    def __init__(self, default_parameter_values=None, energy_units=u.Unit("TeV"), *args, **kwargs):
        if default_parameter_values is None:
            default_parameter_values = {'index':2.5,"cutoff_energy_TeV":1, "phi0": 1e-8, }

        super().__init__(logfunc=self.log_broken_power_law, default_parameter_values=default_parameter_values)