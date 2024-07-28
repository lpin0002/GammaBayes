import numpy as np
from astropy import units as u
import pickle

class GammaBinning(object):
    """ Class for handling data binning and will eventually handle different 
spatial coordinate transforms and pixel sizes for more complex binning schemes.
(Notably WCS coordinate geometry available through Astropy.)
Axis values represent bin centres. Bin edges can be extracted with the 'axis name'_edges (e.g. energy_edges)."""

    def __init__(self, energy_axis, lon_axis, lat_axis):
        self.energy_axis = energy_axis
        self.lon_axis = lon_axis
        self.lat_axis = lat_axis
        self.energy_edges = self._calculate_edges(energy_axis.value, log=True)*energy_axis.unit
        self.lon_edges = self._calculate_edges(lon_axis.value)*lon_axis.unit
        self.lat_edges = self._calculate_edges(lat_axis.value)*lat_axis.unit

        self.num_axes = 3



    @property
    def axes(self):
        return [self.energy_axis, self.lon_axis, self.lat_axis]
    
    @property
    def axes_mesh(self):
        return np.meshgrid(self.energy_axis, self.lon_axis, self.lat_axis, indexing='ij')
    
    @property
    def axes_dim(self):
        return (len(axis) for axis in self.axes)
    
    @property
    def spatial_axes(self):
        return [self.lon_axis, self.lat_axis]



    @classmethod
    def from_params(cls, lon_min, lon_max, lon_bin_size, lat_min, lat_max, lat_bin_size, energy_min, energy_max, energy_bins_per_decade, 
                    energy_unit=u.TeV, angle_unit=u.deg):
        
        lon_min = lon_min * angle_unit if not hasattr(lon_min, 'unit') else lon_min
        lon_max = lon_max * angle_unit if not hasattr(lon_max, 'unit') else lon_max
        lon_bin_size = lon_bin_size * angle_unit if not hasattr(lon_bin_size, 'unit') else lon_bin_size

        lat_min = lat_min * angle_unit if not hasattr(lat_min, 'unit') else lat_min
        lat_max = lat_max * angle_unit if not hasattr(lat_max, 'unit') else lat_max
        lat_bin_size = lat_bin_size * angle_unit if not hasattr(lat_bin_size, 'unit') else lat_bin_size

        energy_min = energy_min * energy_unit if not hasattr(energy_min, 'unit') else energy_min
        energy_max = energy_max * energy_unit if not hasattr(energy_max, 'unit') else energy_max

        lon_bins = np.arange(lon_min.value, lon_max.value + lon_bin_size.value, lon_bin_size.value) * angle_unit
        lat_bins = np.arange(lat_min.value, lat_max.value + lat_bin_size.value, lat_bin_size.value) * angle_unit
        energy_bins = cls.generate_energy_bins(energy_min, energy_max, energy_bins_per_decade, energy_unit)
        return cls(lon_bins, lat_bins, energy_bins)

    @staticmethod
    def generate_energy_bins(energy_min, energy_max, bins_per_decade, energy_unit):
        energy_min_log = np.log10(energy_min.to(energy_unit).value)
        energy_max_log = np.log10(energy_max.to(energy_unit).value)
        num_bins = int((energy_max_log - energy_min_log) * bins_per_decade)
        energy_bins = np.logspace(energy_min_log, energy_max_log, num_bins + 1) * energy_unit
        return energy_bins
    


    def _calculate_edges(self, centers, log=False):
        if log:
            # Logarithmic binning for energy
            log_centers = np.log10(centers)
            log_edges = np.zeros(len(centers) + 1)
            log_edges[1:-1] = (log_centers[:-1] + log_centers[1:]) / 2
            log_edges[0] = log_centers[0] - (log_centers[1] - log_centers[0]) / 2
            log_edges[-1] = log_centers[-1] + (log_centers[-1] - log_centers[-2]) / 2
            return 10**log_edges
        else:
            # Linear binning for spatial axes
            edges = np.zeros(len(centers) + 1)
            edges[1:-1] = (centers[:-1] + centers[1:]) / 2
            edges[0] = centers[0] - (centers[1] - centers[0]) / 2
            edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
            return edges
        

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
    


    def is_subset(self, other):
        return (
            np.all(np.isin(self.energy_axis, other.energy_axis)) and
            np.all(np.isin(self.lon_axis, other.lon_axis)) and
            np.all(np.isin(self.lat_axis, other.lat_axis))
        )

    def is_within_bounds(self, other):
        return (
            self.energy_axis[0] >= other.energy_axis[0] and
            self.energy_axis[-1] <= other.energy_axis[-1] and
            self.lon_axis[0] >= other.lon_axis[0] and
            self.lon_axis[-1] <= other.lon_axis[-1] and
            self.lat_axis[0] >= other.lat_axis[0] and
            self.lat_axis[-1] <= other.lat_axis[-1]
        )

    def __lt__(self, other):
        return self.is_subset(other)

    def __le__(self, other):
        return self.is_subset(other) or self.is_within_bounds(other)

    def __gt__(self, other):
        return other.is_subset(self)

    def __ge__(self, other):
        return other.is_subset(self) or other.is_within_bounds(self)
    

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

