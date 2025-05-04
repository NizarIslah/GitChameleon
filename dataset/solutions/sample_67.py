# library: numpy
# version: 1.21.0
# extra_dependencies: []
import numpy as np


def apply_convolution_valid(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    return np.convolve(arr1, arr2, mode="valid")
