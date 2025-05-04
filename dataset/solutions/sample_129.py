# library: scipy
# version: 1.11.2
# extra_dependencies: []
from scipy.ndimage import rank_filter
import numpy as np


def apply_rank_filter(A: np.ndarray, rank: int, size: int) -> np.ndarray:
    return rank_filter(A, rank, size=size, axes=[1, 2])
