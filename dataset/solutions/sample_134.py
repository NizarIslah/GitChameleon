# library: scipy
# version: 1.9.2
# extra_dependencies: []
from scipy.ndimage import median_filter
import numpy as np


def apply_median_filter(A: np.ndarray, size: int) -> np.ndarray:
    output = np.zeros(A.shape)
    for i in range(A.shape[0]):
        output[i] = median_filter(A[i], size=size)
    return output
