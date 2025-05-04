# library: scipy
# version: 1.11.2
# extra_dependencies: []
from scipy.ndimage import uniform_filter
import numpy as np


def apply_uniform_filter(A: np.ndarray, size: int) -> np.ndarray:
    return uniform_filter(A, size=size, axes=[1, 2])
