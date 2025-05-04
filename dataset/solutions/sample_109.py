# library: scipy
# version: 1.7.3
# extra_dependencies: []
from scipy.spatial import distance
import numpy as np


def compute_wminkowski(
    u: np.ndarray, v: np.ndarray, p: int, w: np.ndarray
) -> np.ndarray:
    return distance.wminkowski(u, v, p=p, w=w)
