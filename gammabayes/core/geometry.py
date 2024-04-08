import numpy as np

class DataGeometry(object):
    """Class to contain metadata relating to a observational event data."""



    def __init__(self, 
                 pixelation_method: str = 'HEALPix', 
                 frame: str = 'galactic',
                 angular_bounds: list | list[list] = [[-3, 3], [-3, 3]], 
                 energy_bounds: list[float] = [1e-1, 1e2], 
                 angular_bins_per_unit: int | list[int] = 10, 
                 energy_bins_per_decade: int = 100, 
                 angular_width: float = None, 
                 angular_range: list | np.ndarray = None,
                 energy_range: list | np.ndarray = None, 
                 **kwargs):
        
        self.pixelation_method = pixelation_method
        self.frame = frame
        self.angular_bounds = angular_bounds
        self.energy_bounds = energy_bounds
        self.angular_bins_per_unit = angular_bins_per_unit
        self.energy_bins_per_decade = energy_bins_per_decade
        self.angular_width = angular_width
        self.angular_range = angular_range
        self.energy_range = energy_range
