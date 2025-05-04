# library: scipy
# version: 1.9.2
# extra_dependencies: []
from scipy.ndimage import gaussian_filter
import numpy as np


def apply_gaussian_filter(A: np.ndarray, sigma: float) -> np.ndarray:
    output = np.zeros(A.shape)
    for i in range(A.shape[0]):
        output[i] = gaussian_filter(A[i], sigma=sigma)
    return output
