# library: scipy
# version: 1.9.2
# extra_dependencies: []
from scipy import linalg
import numpy as np


def compute_matrix_exponential(A: np.ndarray) -> np.ndarray:
    return linalg.expm(A)
