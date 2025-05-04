# library: scipy
# version: 1.9.1
# extra_dependencies: ['numpy==1.21.6']
import warnings
from scipy.linalg import det
import numpy as np

warnings.filterwarnings("error")


def check_invertibility(matrices: np.ndarray) -> np.bool_:
    return np.alltrue([det(A) for A in matrices])
