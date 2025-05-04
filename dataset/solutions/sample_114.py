# library: scipy
# version: 1.9.2
# extra_dependencies: []
from scipy import stats
import numpy as np


def combine_pvalues(A: np.ndarray) -> tuple[float, float]:
    return stats.combine_pvalues(A, "pearson")
