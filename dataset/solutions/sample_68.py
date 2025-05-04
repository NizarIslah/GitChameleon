# library: numpy
# version: 1.21.0
# extra_dependencies: []
import numpy as np


def apply_correlate_full(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.correlate(arr1, arr2, mode="full")
