# library: scipy
# version: 1.9.2
# extra_dependencies: []
from scipy import stats
import numpy as np


def compute_circular_variance(a: np.ndarray) -> float:
    return stats.circvar(a)
