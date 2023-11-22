from .gammapy_wrappers import *
from .irf_normalisation_setup import *
import warnings
try:
    from .load_irf_norms import *
except Exception as e:
    # Print the error message without raising the exception
    print(f"An error occurred: {e}")