# library: scipy
# version: 1.11.2
# extra_dependencies: []
from scipy.ndimage import median_filter
import numpy as np


def apply_median_filter(A: np.ndarray, size: int) -> np.ndarray:
    return median_filter(A, size=size, axes=[1, 2])
