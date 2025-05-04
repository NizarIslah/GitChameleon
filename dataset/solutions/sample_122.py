# library: scipy
# version: 1.11.2
# extra_dependencies: []
from scipy.linalg import det
import numpy as np


def compute_determinant(A: np.ndarray) -> np.ndarray:
    return det(A)
