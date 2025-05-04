# library: scipy
# version: 1.11.1
# extra_dependencies: ['numpy==1.25.1']
import warnings
from scipy.linalg import det
import numpy as np

warnings.filterwarnings("error")


def check_invertibility(matrices: np.ndarray) -> np.bool_:
    return np.all(det(matrices))
