# library: numpy
# version: 1.25.0
# extra_dependencies: []
import numpy as np


def custom_alltrue(arr: np.ndarray) -> np.ndarray:
    return np.all(arr)
