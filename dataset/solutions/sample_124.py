# library: scipy
# version: 1.11.2
# extra_dependencies: []
from scipy.linalg import lu
import numpy as np


def compute_lu_decomposition(
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # return p, l, u
    return lu(A)
