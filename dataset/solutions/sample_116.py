# library: scipy
# version: 1.9.2
# extra_dependencies: []
from scipy import sparse, linalg
import numpy as np


def compute_matrix_exponential(A: sparse.lil_matrix) -> sparse.lil_matrix:
    return sparse.linalg.expm(A)
