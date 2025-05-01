# library: numpy
# version: 1.25.0
# extra_dependencies: []
import numpy as np

def find_common_type(array1:np.ndarray, array2:np.ndarray) -> np.dtype:
    return np.common_type(array1, array2)