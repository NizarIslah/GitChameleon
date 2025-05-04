from typing import Union

# library: scipy
# version: 1.9.2
# extra_dependencies: []
from scipy.ndimage import percentile_filter
import numpy as np


def apply_percentile_filter(
    A: np.ndarray, percentile: Union[int, float], size: int
) -> np.ndarray:
    output = np.zeros(A.shape)
    for i in range(A.shape[0]):
        output[i] = percentile_filter(A[i], percentile=percentile, size=size)
    return output
