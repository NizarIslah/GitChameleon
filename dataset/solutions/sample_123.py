# library: scipy
# version: 1.9.2
# extra_dependencies: []
from scipy.linalg import lu
import numpy as np


def compute_lu_decomposition(
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # return p, l, u
    p, l, u = [np.zeros(A.shape) for i in range(3)]
    for i in range(A.shape[0]):
        p[i], l[i], u[i] = lu(A[i])
    return p, l, u
