import numpy as np
from scipy import interpolate
import pandas as pd
from gammabayes.dark_matter.density_profiles import DM_Profiles
from os import path
ScalarSinglet_Folder_Path = path.dirname(__file__)

from gammabayes.dark_matter.spectral_models.core import DM_ContinuousEmission_Spectrum, CSVDictionary
import time

class Z2_ScalarSinglet(DM_ContinuousEmission_Spectrum):
    """
    Represents the Z2 Scalar Singlet dark matter model, providing mechanisms to calculate the continuous emission spectrum
    based on the dark matter annihilation processes. This class extends the DM_ContinuousEmission_Spectrum class, specializing
    it for the Z2 scalar singlet model by using pre-defined annihilation fractions and parameter axes specific to this model.

    The class utilizes annihilation fractions derived from the DarkSUSY software, formatted and stored in a CSV file. These
    fractions, along with mass and lambda (Higgs coupling constant) axes, are used to initialize the parent class with model-specific
    parameters.

    Attributes:
        darkSUSY_SS_BFs_cleaned_dict (CSVDictionary): A dictionary-like object containing the annihilation fractions and parameter
                                                      axes (mass and lambda) extracted from the cleaned DarkSUSY outputs.

    The initialization process involves:
    - Extracting the annihilation ratios and parameter axes for the Scalar Singlet model.
    - Converting mass units from GeV to TeV to conform with the expected units in the parent class.
    - Removing unnecessary keys from the dictionary to leave only the annihilation fractions.
    - Initializing the parent class with the extracted data, ready for spectral generation.

    Args:
        *args: Variable length argument list passed to the parent class's constructor.
        **kwargs: Arbitrary keyword arguments passed to the parent class's constructor, allowing for the specification of default
                  parameter values and any other necessary initialization options.
    """

    darkSUSY_SS_BFs_cleaned_dict = CSVDictionary(ScalarSinglet_Folder_Path+'/darkSUSY_BFs/darkSUSY_BFs_cleaned.csv', delimiter=' ')

    def __init__(self, *args, **kwargs):
        """
        Initializes the Z2_ScalarSinglet class with model-specific annihilation fractions and parameter axes.

        The initialization process includes the extraction of model parameters from a CSV file containing cleaned outputs
        from the DarkSUSY software, specifically tailored for the Z2 Scalar Singlet dark matter model. The class then
        initializes its parent class, DM_ContinuousEmission_Spectrum, with these parameters to enable spectral calculations.

        Args:
            *args: Variable length argument list for the parent class's constructor.
            **kwargs: Arbitrary keyword arguments for the parent class's constructor, including options to specify
                      default parameter values such as mass and lambda.
        """

        # Extracting the annihilation ratios for the Scalar Singlet model
        self.mass_axis, self.lambda_axis = self.darkSUSY_SS_BFs_cleaned_dict['mS [GeV]']/1e3, self.darkSUSY_SS_BFs_cleaned_dict['lahS']
        
        # Also as you can probably tell, darkSUSY output the masses in GeV, so we divide by 1e3 to get them in TeV. If you are also 
            # implementing your own class it is currently required to be over TeV mass/energy

        self.darkSUSY_SS_BFs_cleaned_dict.pop('mS [GeV]')
        self.darkSUSY_SS_BFs_cleaned_dict.pop('lahS')

        super().__init__(annihilation_fractions = self.darkSUSY_SS_BFs_cleaned_dict, 
                         parameter_interpolation_values = [self.mass_axis, self.lambda_axis],
                         default_parameter_values={'mass':1.0, 'lahS':0.1},
                         *args, **kwargs)
        





