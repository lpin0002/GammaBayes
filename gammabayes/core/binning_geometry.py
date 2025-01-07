import numpy as np
from astropy import units as u
import pickle

class GammaBinning:
    """
    Class for handling data binning for gamma-ray astronomy.
    This class can handle different spatial coordinate transforms and pixel sizes
    for more complex binning schemes, such as WCS coordinate geometry via Astropy.

    Attributes:
        energy_axis (Quantity): Bin centers for the energy axis.
        lon_axis (Quantity): Bin centers for the longitude axis.
        lat_axis (Quantity): Bin centers for the latitude axis.
        energy_edges (Quantity): Bin edges for the energy axis.
        lon_edges (Quantity): Bin edges for the longitude axis.
        lat_edges (Quantity): Bin edges for the latitude axis.
        num_axes (int): Number of axes (always 3: energy, longitude, latitude).
    """

    def __init__(self, energy_axis, lon_axis, lat_axis):
        """
        Initialize the GammaBinning instance with axis values.

        Args:
            energy_axis (Quantity): Bin centers for the energy axis.
            lon_axis (Quantity): Bin centers for the longitude axis.
            lat_axis (Quantity): Bin centers for the latitude axis.
        """
        self.energy_axis = energy_axis
        self.lon_axis = lon_axis
        self.lat_axis = lat_axis

        # Calculate bin edges for all axes
        self.energy_edges = self._calculate_edges(energy_axis.value, log=True) * energy_axis.unit
        self.lon_edges = self._calculate_edges(lon_axis.value) * lon_axis.unit
        self.lat_edges = self._calculate_edges(lat_axis.value) * lat_axis.unit

        self.num_axes = 3  # Fixed number of axes

    @property
    def axes(self):
        """List of all axes (energy, longitude, latitude)."""
        return [self.energy_axis, self.lon_axis, self.lat_axis]

    @property
    def lon_res(self):
        """Resolution of the longitude axis (difference between consecutive bin centers)."""
        return np.diff(self.lon_axis)[0]

    @property
    def lat_res(self):
        """Resolution of the latitude axis (difference between consecutive bin centers)."""
        return np.diff(self.lat_axis)[0]

    @property
    def axes_mesh(self):
        """Meshgrid of all axes (energy, longitude, latitude) with 'ij' indexing."""
        return np.meshgrid(self.energy_axis, self.lon_axis, self.lat_axis, indexing='ij')

    @property
    def axes_dim(self):
        """Tuple of the lengths of each axis."""
        return (*(len(axis) for axis in self.axes), )

    @property
    def spatial_axes(self):
        """List of spatial axes (longitude and latitude)."""
        return [self.lon_axis, self.lat_axis]

    @property
    def spatial_centre(self):
        """Spatial center (mean of longitude and latitude axes)."""
        return np.asarray([np.mean(self.lon_axis.value), np.mean(self.lat_axis.value)]) * self.lon_axis.unit

    @classmethod
    def from_params(cls, lon_min, lon_max, lon_bin_size, lat_min, lat_max, lat_bin_size, energy_min, energy_max, energy_bins_per_decade, 
                    energy_unit=u.TeV, angle_unit=u.deg):
        """
        Create a GammaBinning instance from axis parameters.

        Args:
            lon_min, lon_max (float or Quantity): Longitude range.
            lon_bin_size (float or Quantity): Longitude bin size.
            lat_min, lat_max (float or Quantity): Latitude range.
            lat_bin_size (float or Quantity): Latitude bin size.
            energy_min, energy_max (float or Quantity): Energy range.
            energy_bins_per_decade (int): Number of energy bins per decade.
            energy_unit (Unit, optional): Unit of energy. Defaults to u.TeV.
            angle_unit (Unit, optional): Unit of angles. Defaults to u.deg.

        Returns:
            GammaBinning: Initialized instance.
        """
        energy_unit = u.Unit(energy_unit)
        angle_unit = u.Unit(angle_unit)

        # Ensure all inputs have appropriate units
        lon_min = lon_min * angle_unit if not hasattr(lon_min, 'unit') else lon_min
        lon_max = lon_max * angle_unit if not hasattr(lon_max, 'unit') else lon_max
        lon_bin_size = lon_bin_size * angle_unit if not hasattr(lon_bin_size, 'unit') else lon_bin_size

        lat_min = lat_min * angle_unit if not hasattr(lat_min, 'unit') else lat_min
        lat_max = lat_max * angle_unit if not hasattr(lat_max, 'unit') else lat_max
        lat_bin_size = lat_bin_size * angle_unit if not hasattr(lat_bin_size, 'unit') else lat_bin_size

        energy_min = energy_min * energy_unit if not hasattr(energy_min, 'unit') else energy_min
        energy_max = energy_max * energy_unit if not hasattr(energy_max, 'unit') else energy_max

        # Generate bins for each axis
        lon_bins = np.arange(lon_min.value, lon_max.value + lon_bin_size.value, lon_bin_size.value) * angle_unit
        lat_bins = np.arange(lat_min.value, lat_max.value + lat_bin_size.value, lat_bin_size.value) * angle_unit
        energy_bins = cls.generate_energy_bins(energy_min, energy_max, energy_bins_per_decade, energy_unit)
        return cls(energy_bins, lon_bins, lat_bins)

    @staticmethod
    def generate_energy_bins(energy_min, energy_max, bins_per_decade, energy_unit):
        """
        Generate logarithmic energy bins.

        Args:
            energy_min, energy_max (Quantity): Energy range.
            bins_per_decade (int): Number of bins per decade.
            energy_unit (Unit): Unit of energy.

        Returns:
            Quantity: Energy bin edges.
        """
        energy_min_log = np.log10(energy_min.to(energy_unit).value)
        energy_max_log = np.log10(energy_max.to(energy_unit).value)
        num_bins = int((energy_max_log - energy_min_log) * bins_per_decade)
        energy_bins = np.logspace(energy_min_log, energy_max_log, num_bins + 1) * energy_unit
        return energy_bins

    def _calculate_edges(self, centers, log=False):
        """
        Calculate bin edges from bin centers.

        Args:
            centers (ndarray): Bin centers.
            log (bool, optional): Whether to calculate logarithmic edges. Defaults to False.

        Returns:
            ndarray: Bin edges.
        """
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
        """
        Check equality with another GammaBinning instance.

        Args:
            other (GammaBinning): Another instance to compare.

        Returns:
            bool: True if equal, False otherwise.
        """
        if not isinstance(other, GammaBinning):
            return False
        return (
            np.array_equal(self.energy_axis, other.energy_axis.to(self.energy_axis.unit)) and
            np.array_equal(self.lon_axis, other.lon_axis.to(self.lon_axis.unit)) and
            np.array_equal(self.lat_axis, other.lat_axis.to(self.lat_axis.unit))
        )

    def __ne__(self, other):
        """Check inequality with another GammaBinning instance."""
        return not self.__eq__(other)

    @classmethod
    def axes_has_same_bounds(cls, axis1, axis2):
        """
        Check if two axes have the same bounds.

        Args:
            axis1, axis2 (Quantity): Axes to compare.

        Returns:
            bool: True if bounds are the same, False otherwise.
        """
        return np.isclose(axis1[0].value, axis2[0].to(axis1.unit).value) and np.isclose(axis1[-1].value, axis2[-1].to(axis1.unit).value)

    def has_same_bounds(self, other):
        """
        Check if all axes have the same bounds as another GammaBinning instance.

        Args:
            other (GammaBinning): Another instance to compare.

        Returns:
            bool: True if bounds are the same, False otherwise.
        """
        return (self.axes_has_same_bounds(self.energy_axis, other.energy_axis) and
                self.axes_has_same_bounds(self.lon_axis, other.lon_axis) and
                self.axes_has_same_bounds(self.lat_axis, other.lat_axis))

    @classmethod
    def axis_is_within_bounds(cls, axis1, axis2):
        """
        Check if an axis is within the bounds of another axis.

        Args:
            axis1, axis2 (Quantity): Axes to compare.

        Returns:
            bool: True if axis1 is within bounds of axis2, False otherwise.
        """
        return axis1[0].value > axis2[0].to(axis1.unit).value and axis1[-1].value < axis2[-1].to(axis1.unit).value

    def is_within_bounds(self, other):
        """
        Check if all axes are within the bounds of another GammaBinning instance.

        Args:
            other (GammaBinning): Another instance to compare.

        Returns:
            bool: True if within bounds, False otherwise.
        """
        return (self.axis_is_within_bounds(self.energy_axis, other.energy_axis) and
                self.axis_is_within_bounds(self.lon_axis, other.lon_axis) and
                self.axis_is_within_bounds(self.lat_axis, other.lat_axis))

    def __lt__(self, other):
        """Check if this instance is strictly within bounds of another."""
        return self.is_within_bounds(other)

    def __le__(self, other):
        """Check if this instance is within or has the same bounds as another."""
        return self.has_same_bounds(other) or self.is_within_bounds(other)

    def __gt__(self, other):
        """Check if another instance is strictly within bounds of this instance."""
        return other.is_within_bounds(self)

    def __ge__(self, other):
        """Check if another instance is within or has the same bounds as this instance."""
        return other.has_same_bounds(self) or other.is_within_bounds(self)

    def to_dict(self):
        """Convert the GammaBinning instance to a dictionary."""
        return {
            "energy_axis": self.energy_axis,
            "lon_axis": self.lon_axis,
            "lat_axis": self.lat_axis,
        }

    def save(self, filename):
        """Save the GammaBinning instance to a file using pickle.

        Args:
            filename (str): Path to the file to save.
        """
        data_dict = self.to_dict()
        with open(filename, 'wb') as f:
            pickle.dump(data_dict, f)

    @classmethod
    def from_dict(cls, data_dict):
        """Create a GammaBinning instance from a dictionary.

        Args:
            data_dict (dict): Dictionary containing axis data.

        Returns:
            GammaBinning: Initialized instance.
        """
        return cls(energy_axis=data_dict["energy_axis"], 
                   lon_axis=data_dict["lon_axis"], 
                   lat_axis=data_dict["lat_axis"])

    @classmethod
    def load(cls, filename):
        """Load a GammaBinning instance from a file.

        Args:
            filename (str): Path to the file to load.

        Returns:
            GammaBinning: Loaded instance.
        """
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
        return cls.from_dict(data_dict)
