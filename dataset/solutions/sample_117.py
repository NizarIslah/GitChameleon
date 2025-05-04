# library: scipy
# version: 1.8.1
# extra_dependencies: []
from scipy import stats
import numpy as np


def compute_circular_variance(a: np.ndarray) -> float:
    return 1 - np.abs(np.mean(np.exp(1j * a)))
