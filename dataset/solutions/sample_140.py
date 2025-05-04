# library: scipy
# version: 1.9.2
# extra_dependencies: []
from scipy.ndimage import maximum_filter
import numpy as np


def apply_maximum_filter(A: np.ndarray, size: int) -> np.ndarray:
    output = np.zeros(A.shape)
    for i in range(A.shape[0]):
        output[i] = maximum_filter(A[i], size=size)
    return output
