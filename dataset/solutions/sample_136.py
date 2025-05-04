# library: scipy
# version: 1.9.2
# extra_dependencies: []
from scipy.ndimage import uniform_filter
import numpy as np


def apply_uniform_filter(A: np.ndarray, size: int) -> np.ndarray:
    output = np.zeros(A.shape)
    for i in range(A.shape[0]):
        output[i] = uniform_filter(A[i], size=size)
    return output
