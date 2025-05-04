# library: scipy
# version: 1.8.1
# extra_dependencies: []
from scipy import stats
import numpy as np


def combine_pvalues(A: np.ndarray) -> tuple[float, float]:
    output = stats.combine_pvalues(A, "pearson")
    return (-output[0], 1 - output[1])
