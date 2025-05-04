from typing import Union

# library: scipy
# version: 1.11.2
# extra_dependencies: []
from scipy.ndimage import percentile_filter
import numpy as np


def apply_percentile_filter(
    A: np.ndarray, percentile: Union[int, float], size: int
) -> np.ndarray:
    return percentile_filter(A, percentile=percentile, size=size, axes=[1, 2])
