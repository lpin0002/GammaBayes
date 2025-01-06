
try:
    from jax import numpy as np

except Exception as err:
    import numpy as np
    print(__file__, err)

import numpy
from numpy import ndarray


import pickle

class GammaBinning(object):
    """ Class for handling data binning and will eventually handle different 
spatial coordinate transforms and pixel sizes for more complex binning schemes.
(Notably WCS coordinate geometry available through Astropy.)
Axis values represent bin centres. Bin edges can be extracted with the 'axis name'_edges (e.g. energy_edges)."""

    def __init__(self, energy_axis, lon_axis, lat_axis):
        self.energy_axis = np.asarray(energy_axis)
        self.lon_axis = np.asarray(lon_axis)
        self.lat_axis = np.asarray(lat_axis)



        self.energy_edges = self._calculate_edges(energy_axis, log=True)
        self.lon_edges = self._calculate_edges(lon_axis)
        self.lat_edges = self._calculate_edges(lat_axis)

        self.num_axes = 3



    @property
    def axes(self):
        return [self.energy_axis, self.lon_axis, self.lat_axis]
    
    @property
    def lon_res(self):
        return np.diff(self.lon_axis)[0]
    
    @property
    def lat_res(self):
        return np.diff(self.lat_axis)[0]
    
    @property
    def axes_mesh(self):
        return np.meshgrid(self.energy_axis, self.lon_axis, self.lat_axis, indexing='ij')
    
    @property
    def axes_dim(self):
        return (*(len(axis) for axis in self.axes), )
    
    @property
    def spatial_axes(self):
        return [self.lon_axis, self.lat_axis]
    

    @property
    def spatial_centre(self):
        return np.asarray([np.mean(self.lon_axis), np.mean(self.lat_axis)])
    


    @classmethod
    def from_params(cls, lon_min, lon_max, lon_bin_size, lat_min, lat_max, lat_bin_size, energy_min, energy_max, energy_bins_per_decade):
        
        lon_bins = np.arange(lon_min, lon_max + lon_bin_size, lon_bin_size)
        lat_bins = np.arange(lat_min, lat_max + lat_bin_size, lat_bin_size)
        energy_bins = cls.generate_energy_bins(energy_min, energy_max, energy_bins_per_decade)
        return cls(energy_bins, lon_bins, lat_bins)

    @staticmethod
    def generate_energy_bins(energy_min, energy_max, bins_per_decade):
        energy_min_log = np.log10(energy_min)
        energy_max_log = np.log10(energy_max)
        num_bins = int((energy_max_log - energy_min_log) * bins_per_decade)
        energy_bins = np.logspace(energy_min_log, energy_max_log, num_bins + 1)
        return energy_bins
    


    def _calculate_edges(self, centers, log=False):
        if log:
            # Logarithmic binning for energy
            log_centers = numpy.log10(centers)
            log_edges = numpy.zeros(len(centers) + 1)
            log_edges[1:-1] = (log_centers[:-1] + log_centers[1:]) / 2
            log_edges[0] = log_centers[0] - (log_centers[1] - log_centers[0]) / 2
            log_edges[-1] = log_centers[-1] + (log_centers[-1] - log_centers[-2]) / 2
            return np.asarray(10**log_edges)
        else:
            # Linear binning for spatial axes
            edges = numpy.zeros(len(centers) + 1)
            edges[1:-1] = (centers[:-1] + centers[1:]) / 2
            edges[0] = centers[0] - (centers[1] - centers[0]) / 2
            edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
            return np.asarray(edges)
        

    def __eq__(self, other):
        if not isinstance(other, GammaBinning):
            return False
        return (
            np.array_equal(self.energy_axis, other.energy_axis) and
            np.array_equal(self.lon_axis, other.lon_axis) and
            np.array_equal(self.lat_axis, other.lat_axis)
        )

    def __ne__(self, other):
        return not self.__eq__(other)
    

    @classmethod
    def axes_has_same_bounds(self, axis1, axis2):
        return np.isclose(axis1[0], axis2[0]) and np.isclose(axis1[-1], axis2[-1])



    def has_same_bounds(self, other):
        return (self.axes_has_same_bounds(self.energy_axis, other.energy_axis) and
                self.axes_has_same_bounds(self.lon_axis, other.lon_axis) and
                self.axes_has_same_bounds(self.lat_axis, other.lat_axis))
    
    @classmethod
    def axis_is_within_bounds(self, axis1, axis2):
        return    axis1[0] > axis2[0] and axis1[-1] < axis2[-1]


    def is_within_bounds(self, other):
        return (self.axis_is_within_bounds(self.energy_axis, other.energy_axis) and
                self.axis_is_within_bounds(self.lon_axis, other.lon_axis) and
                self.axis_is_within_bounds(self.lat_axis, other.lat_axis)
                )

    def __lt__(self, other):
        return self.is_within_bounds(other)

    def __le__(self, other):
        return self.has_same_bounds(other) or self.is_within_bounds(other)

    def __gt__(self, other):
        return other.is_within_bounds(self)

    def __ge__(self, other):
        return other.has_same_bounds(self) or other.is_within_bounds(self)
    

    def to_dict(self):
        return {
            "energy_axis": self.energy_axis,
            "lon_axis": self.lon_axis,
            "lat_axis": self.lat_axis,
            }
    

    def save(self, filename):
        """Save the GammaBinning instance to a file using pickle."""
        data_dict = self.to_dict()
        with open(filename, 'wb') as f:
            pickle.dump(data_dict, f)

    @classmethod
    def from_dict(cls, data_dict):
        return cls(energy_axis=data_dict["energy_axis"], 
                        lon_axis=data_dict["lon_axis"], 
                        lat_axis=data_dict["lat_axis"])
    
    @classmethod
    def load(cls, filename):
        """Load a GammaBinning instance from a file."""
        
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
        

        return cls.from_dict(data_dict)

