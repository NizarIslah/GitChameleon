# library: scipy
# version: 1.11.2
# extra_dependencies: []
from scipy.ndimage import minimum_filter
import numpy as np


def apply_minimum_filter(A: np.ndarray, size: int) -> np.ndarray:
    return minimum_filter(A, size=size, axes=[1, 2])
