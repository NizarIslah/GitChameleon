# library: scipy
# version: 1.11.1
# extra_dependencies: ['numpy==1.25.1']
import numpy as np
from scipy.stats import hmean


def count_unique_hmean(data: np.ndarray) -> int:
    # data shape: (n, m)
    # n: number of arrays
    # m: number of elements in each array
    hmean_values = hmean(np.asarray(data), axis=1)
    unique_vals = np.unique(hmean_values, equal_nan=False).shape[0]
    return unique_vals
