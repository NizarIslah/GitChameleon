# library: scipy
# version: 1.10.1
# extra_dependencies: []
from scipy.ndimage import gaussian_filter1d
import numpy as np


def apply_gaussian_filter1d(x: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    return gaussian_filter1d(x, radius=radius, sigma=sigma)
