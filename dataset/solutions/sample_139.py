# library: scipy
# version: 1.11.2
# extra_dependencies: []
from scipy.ndimage import maximum_filter
import numpy as np


def apply_maximum_filter(A: np.ndarray, size: int) -> np.ndarray:
    return maximum_filter(A, size=size, axes=[1, 2])
