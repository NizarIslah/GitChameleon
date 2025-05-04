# library: numpy
# version: 1.25.0
# extra_dependencies: []
import numpy as np


def find_common_type(arr1: np.ndarray, arr2: np.ndarray) -> np.dtype:
    return np.common_type(arr1, arr2)
