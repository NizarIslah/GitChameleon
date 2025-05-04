# library: scipy
# version: 1.8.1
# extra_dependencies: []
from scipy import sparse, linalg
import numpy as np


def compute_matrix_exponential(A: sparse.lil_matrix) -> sparse.lil_matrix:
    return linalg.expm(A)
