# library: scipy
# version: 1.9.2
# extra_dependencies: []
from scipy.ndimage import rank_filter
import numpy as np


def apply_rank_filter(A: np.ndarray, rank: int, size: int) -> np.ndarray:
    output = np.zeros(A.shape)
    for i in range(A.shape[0]):
        output[i] = rank_filter(A[i], rank, size=size)
    return output
