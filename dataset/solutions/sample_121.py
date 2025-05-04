# library: scipy
# version: 1.9.2
# extra_dependencies: []
from scipy.linalg import det
import numpy as np


def compute_determinant(A: np.ndarray) -> np.ndarray:
    output = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        output[i] = det(A[i])
    return output
