# library: scipy
# version: 1.11.2
# extra_dependencies: []
from scipy.ndimage import gaussian_filter
import numpy as np


def apply_gaussian_filter(A: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(A, sigma=sigma, axes=[1, 2])
