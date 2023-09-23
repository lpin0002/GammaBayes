import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u


from gammapy.data import DataStore
from gammapy.maps import MapAxis
from gammapy.makers.utils import make_theta_squared_table
from gammapy.visualization import plot_theta_squared_table

data_store = DataStore.from_dir("gammapy-extra/datasets/hess-dl3-dr1")

print(data_store.info())

obs = data_store.get_observations([20136])

obs.peek()