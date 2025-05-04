# library: scipy
# version: 1.8.1
# extra_dependencies: []
from scipy import linalg
import numpy as np


def compute_matrix_exponential(A: np.ndarray) -> np.ndarray:
    return np.stack([linalg.expm(A[i]) for i in range(A.shape[0])], axis=0)
