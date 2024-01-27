import numpy as np


def bin_centres_to_edges(axis: np.ndarray) -> np.ndarray:
    return np.append(axis-np.diff(axis)[0]/2, axis[-1]+np.diff(axis)[0]/2)